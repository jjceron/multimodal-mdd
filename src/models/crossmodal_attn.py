"""Cross-modal attention fusion for EEG + audio."""

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.scale


# --- Removed: WindowClassifier, LowRankLinear, Adapter ---


class AttentionPool(nn.Module):
    """Learnable weighted sum over node embeddings."""

    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim, 1))

    def forward(self, x):
        scores = x @ self.w / math.sqrt(x.shape[-1])
        weights = torch.softmax(scores, dim=1)
        return (x * weights).sum(dim=1)


class CrossModalFusion(nn.Module):
    """Set-to-set bidirectional cross-attention: EEG [K_e] ↔ Audio [K_a].

    EEG queries attend to Audio keys/values → [B, K_e, H] output.
    Audio queries attend to EEG keys/values → [B, K_a, H] output.
    No positional pairing — full K_e × K_a attention matrix.
    """

    def __init__(self, dim, n_heads=2, attn_dropout=0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.scale = self.d_head**-0.5
        self.attn_drop = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()

        self.e_q = nn.Linear(dim, dim, bias=False)
        self.e_k = nn.Linear(dim, dim, bias=False)
        self.e_v = nn.Linear(dim, dim, bias=False)
        self.e_o = nn.Linear(dim, dim, bias=False)
        self.a_q = nn.Linear(dim, dim, bias=False)
        self.a_k = nn.Linear(dim, dim, bias=False)
        self.a_v = nn.Linear(dim, dim, bias=False)
        self.a_o = nn.Linear(dim, dim, bias=False)

        self.norm_e = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)

    def _attend(self, q, k, v, o_proj, mask_kv=None):
        B, L_q = q.shape[0], q.shape[1]
        L_k = k.shape[1]
        q = q.view(B, L_q, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L_k, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L_k, self.n_heads, self.d_head).transpose(1, 2)
        attn = q @ k.transpose(-2, -1) * self.scale
        if mask_kv is not None:
            attn = attn.masked_fill(mask_kv.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L_q, self.dim)
        return o_proj(out), attn.detach()

    def forward(self, eeg, audio, mask_eeg=None, mask_audio=None):
        e_out, e_attn = self._attend(self.e_q(eeg), self.a_k(audio), self.a_v(audio), self.e_o, mask_audio)
        eeg = self.norm_e(eeg + e_out)
        a_out, a_attn = self._attend(self.a_q(audio), self.e_k(eeg), self.e_v(eeg), self.a_o, mask_eeg)
        audio = self.norm_a(audio + a_out)
        self._attn_entropy = torch.tensor(0.0)
        return eeg, audio, (e_attn, a_attn)


class SelfAttentionBlock(nn.Module):
    """Standard transformer encoder block (pre-norm)."""

    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        x = (
            x
            + self.attn(
                self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=mask
            )[0]
        )
        x = x + self.mlp(self.norm2(x))
        return x


class CrossModalAttention(nn.Module):
    """Cross-modal fusion with set-to-set cross-attention.

    Each modality maintains its own window count (K_eeg, K_audio).
    Cross-attention: EEG [K_e] queries × Audio [K_a] keys (and vice versa).
    Pool per modality → fuse → classify.
    """

    def __init__(
        self,
        eeg_dim,
        aud_dim,
        hidden=32,
        n_heads=2,
        n_self_attn_layers=0,
        self_attn_heads=4,
        self_attn_dropout=0.1,
        fusion="cross_attn",
        pooling="mean",
        dropout=0.5,
        feat_dropout=0.0,
        attn_dropout=0.0,
        n_cross_layers=1,
    ):
        super().__init__()
        self.fusion = fusion
        self.pooling = pooling
        self.hidden = hidden
        self.n_cross_layers = n_cross_layers

        self.feat_drop = nn.Dropout(feat_dropout) if feat_dropout > 0 else nn.Identity()

        self.eeg_proj = nn.Linear(eeg_dim, hidden)
        self.aud_proj = nn.Linear(aud_dim, hidden)
        self.eeg_rms = RMSNorm(hidden)
        self.aud_rms = RMSNorm(hidden)

        if fusion == "cross_attn":
            self.cross_layers = nn.ModuleList([
                CrossModalFusion(hidden, n_heads, attn_dropout=attn_dropout)
                for _ in range(n_cross_layers)
            ])

        self.self_attn_layers = nn.ModuleList([
            SelfAttentionBlock(hidden, self_attn_heads, self_attn_dropout)
            for _ in range(n_self_attn_layers)
        ])

        if pooling == "attn":
            self.pool = AttentionPool(hidden)

        self.fusion_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden, 1)

        self._attn_weights = None
        self._attn_entropy = torch.tensor(0.0)

    def _pool(self, z, mask):
        if self.pooling == "mean":
            if mask is not None:
                n = mask.sum(dim=1, keepdim=True).clamp(min=1)
                return (z * mask.unsqueeze(-1)).sum(dim=1) / n
            return z.mean(dim=1)
        if self.pooling == "attn":
            if mask is not None:
                z = z * mask.unsqueeze(-1)
            return self.pool(z)
        return z.mean(dim=1)

    def forward(self, z_eeg, z_audio, mask_eeg=None, mask_audio=None, return_window=False):
        e = self.eeg_rms(self.eeg_proj(z_eeg))
        a = self.aud_rms(self.aud_proj(z_audio))
        if self.training:
            e = self.feat_drop(e)
            a = self.feat_drop(a)

        attn_mask_e = (mask_eeg == 0) if mask_eeg is not None else None
        attn_mask_a = (mask_audio == 0) if mask_audio is not None else None
        for layer in self.self_attn_layers:
            e = layer(e, mask=attn_mask_e)
            a = layer(a, mask=attn_mask_a)

        if self.fusion == "cross_attn":
            for i, layer in enumerate(self.cross_layers):
                e_out, a_out, attn_w = layer(e, a, mask_eeg, mask_audio)
                if i < len(self.cross_layers) - 1:
                    e, a = e_out, a_out
                else:
                    e, a = e_out, a_out
                    self._attn_weights = attn_w

        win_logits = None
        if return_window:
            win_logits = self.head(e).squeeze(-1)
            if mask_eeg is not None:
                win_logits = win_logits.masked_fill(mask_eeg == 0, float("-inf"))

        e_pool = self._pool(e, mask_eeg)
        a_pool = self._pool(a, mask_audio)
        z = self.fusion_drop((e_pool + a_pool) / 2)
        logits = self.head(z).squeeze(-1)

        if return_window:
            return logits, None, win_logits
        return logits

    def forward_per_window(self, z_eeg, z_audio, mask_eeg=None, mask_audio=None):
        e = self.eeg_rms(self.eeg_proj(z_eeg))
        a = self.aud_rms(self.aud_proj(z_audio))

        attn_mask_e = (mask_eeg == 0) if mask_eeg is not None else None
        attn_mask_a = (mask_audio == 0) if mask_audio is not None else None
        for layer in self.self_attn_layers:
            e = layer(e, mask=attn_mask_e)
            a = layer(a, mask=attn_mask_a)

        if self.fusion == "cross_attn":
            for layer in self.cross_layers:
                e, a, _ = layer(e, a, mask_eeg, mask_audio)

        win_logits = self.head(e).squeeze(-1)
        if mask_eeg is not None:
            win_logits = win_logits.masked_fill(mask_eeg == 0, float("-inf"))
        return win_logits


class PerceiverCrossModal(nn.Module):
    """Cross-modal fusion using Perceiver-style attention with learnable latents.

    Uses a small set of learnable latent tokens that sequentially cross-attend
    to EEG features, then Audio features, then self-attend among themselves.
    This creates a parameter-efficient bottleneck that compresses multi-modal
    information into a compact representation.

    Compatible with the existing training pipeline:
    - forward(z_eeg, z_audio, mask, return_window) for training
    - forward_per_window(z_eeg, z_audio, mask) for majority voting
    - _attn_entropy for entropy regularization
    """

    def __init__(
        self,
        eeg_dim,
        aud_dim,
        hidden=32,
        n_heads=2,
        dropout=0.5,
        feat_dropout=0.0,
        attn_dropout=0.0,
        n_cross_layers=1,
        **kwargs,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_cross_layers = n_cross_layers

        # Project to hidden dimension
        self.eeg_proj = nn.Linear(eeg_dim, hidden)
        self.aud_proj = nn.Linear(aud_dim, hidden)
        self.eeg_norm = RMSNorm(hidden)
        self.aud_norm = RMSNorm(hidden)

        # Feature dropout
        self.feat_drop = nn.Dropout(feat_dropout) if feat_dropout > 0 else nn.Identity()

        # Learnable latent tokens
        n_latents = 4
        self.latents = nn.Parameter(torch.randn(1, n_latents, hidden))

        # Cross-attention: latents (Q) attend to modality features (K, V)
        self.cross_attn_eeg = nn.MultiheadAttention(
            hidden, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.cross_attn_aud = nn.MultiheadAttention(
            hidden, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.cross_norm_eeg = nn.LayerNorm(hidden)
        self.cross_norm_aud = nn.LayerNorm(hidden)

        # Self-attention among latents (standard transformer block)
        self.self_attn = nn.MultiheadAttention(
            hidden, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.self_norm1 = nn.LayerNorm(hidden)
        self.self_mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.Dropout(dropout),
        )
        self.self_norm2 = nn.LayerNorm(hidden)

        # Per-window auxiliary head (for majority voting evaluation)
        self.win_head = nn.Linear(hidden * 2, 1)

        # Classifier
        self.fusion_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden, 1)

        # Compatibility attributes for the training loop
        self._attn_entropy = torch.tensor(0.0)
        self._contrast_loss = None
        self._attn_weights = None
        self._window_pool_weights = None

    def _encode_project(self, z_eeg, z_audio):
        """Projection → dropout."""
        e = self.eeg_norm(self.eeg_proj(z_eeg))
        a = self.aud_norm(self.aud_proj(z_audio))
        e = self.feat_drop(e)
        a = self.feat_drop(a)
        return e, a

    def _perceiver_forward(self, e, a, mask_eeg, mask_audio):
        """Latent tokens → cross-attend EEG → cross-attend Audio → self-attend."""
        B = e.shape[0]
        latents = self.latents.expand(B, -1, -1)
        kp_e = ~mask_eeg.bool() if mask_eeg is not None else None
        kp_a = ~mask_audio.bool() if mask_audio is not None else None

        for _ in range(self.n_cross_layers):
            l_e, _ = self.cross_attn_eeg(latents, e, e, key_padding_mask=kp_e)
            latents = self.cross_norm_eeg(latents + l_e)
            l_a, _ = self.cross_attn_aud(latents, a, a, key_padding_mask=kp_a)
            latents = self.cross_norm_aud(latents + l_a)

            l_sa, _ = self.self_attn(latents, latents, latents)
            latents = self.self_norm1(latents + l_sa)
            latents = self.self_norm2(latents + self.self_mlp(latents))

        self._attn_entropy = torch.tensor(0.0)
        return latents.mean(dim=1)

    def forward(self, z_eeg, z_audio, mask_eeg=None, mask_audio=None, return_window=False):
        e, a = self._encode_project(z_eeg, z_audio)

        z = self._perceiver_forward(e, a, mask_eeg, mask_audio)

        win_logits_post = None
        if return_window:
            win_feats = torch.cat([e, a], dim=-1)
            win_logits_post = self.win_head(win_feats).squeeze(-1)

        z = self.fusion_drop(z)
        logits = self.head(z).squeeze(-1)

        if return_window:
            return logits, None, win_logits_post
        return logits

    def forward_per_window(self, z_eeg, z_audio, mask_eeg=None, mask_audio=None):
        """Per-window logits via simple head on projected features (for majority vote)."""
        e, a = self._encode_project(z_eeg, z_audio)
        win_feats = torch.cat([e, a], dim=-1)
        win_logits = self.win_head(win_feats).squeeze(-1)
        ec_mask = mask_eeg if mask_eeg is not None else mask_audio
        if ec_mask is not None:
            win_logits = win_logits.masked_fill(ec_mask == 0, float("-inf"))
        return win_logits


class LinearProbe(nn.Module):
    """Linear probe baseline: pool EEG + Audio → concat → Linear → logit."""

    def __init__(self, eeg_dim, aud_dim, **kwargs):
        super().__init__()
        self.head = nn.Linear(eeg_dim + aud_dim, 1)
        self._attn_entropy = torch.tensor(0.0)
        self._attn_weights = None

    def _pool(self, z, mask):
        if mask is not None:
            n = mask.sum(dim=1, keepdim=True).clamp(min=1)
            return (z * mask.unsqueeze(-1)).sum(dim=1) / n
        return z.mean(dim=1)

    def forward(self, z_eeg, z_audio, mask_eeg=None, mask_audio=None, return_window=False):
        e = self._pool(z_eeg, mask_eeg)
        a = self._pool(z_audio, mask_audio)
        h = torch.cat([e, a], dim=-1)
        logits = self.head(h).squeeze(-1)

        if return_window:
            win = torch.cat([z_eeg, z_audio], dim=-1)
            win_logits_post = self.head(win).squeeze(-1)
            return logits, None, win_logits_post
        return logits

    def forward_per_window(self, z_eeg, z_audio, mask_eeg=None, mask_audio=None):
        win = torch.cat([z_eeg, z_audio], dim=-1)
        win_logits = self.head(win).squeeze(-1)
        ec_mask = mask_eeg if mask_eeg is not None else mask_audio
        if ec_mask is not None:
            win_logits = win_logits.masked_fill(ec_mask == 0, float("-inf"))
        return win_logits


class GatedCrossModalFusion(nn.Module):
    """Gated cross-modal fusion with element-wise interaction.

    Projects each modality to shared hidden space, then:
    1. Learned gating (sigmoid) to weight modalities per subject
    2. Hadamard product for cross-modal interaction
    3. Concatenate gated sum + cross → project → classify
    """

    def __init__(self, eeg_dim, aud_dim, hidden=48, dropout=0.3, **kwargs):
        super().__init__()
        self.hidden = hidden

        self.e_proj = nn.Sequential(
            nn.Linear(eeg_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
        )
        self.a_proj = nn.Sequential(
            nn.Linear(aud_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
        )

        self.gate = nn.Linear(hidden * 2, hidden)
        self.fusion_proj = nn.Linear(hidden * 2, hidden)
        self.fusion_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden, 1)

        self._attn_entropy = torch.tensor(0.0)
        self._attn_weights = None

    def _pool(self, z, mask):
        if mask is not None:
            n = mask.sum(dim=1, keepdim=True).clamp(min=1)
            return (z * mask.unsqueeze(-1)).sum(dim=1) / n
        return z.mean(dim=1)

    def _fuse(self, e, a):
        g = torch.sigmoid(self.gate(torch.cat([e, a], dim=-1)))
        fused = g * e + (1 - g) * a
        cross = e * a
        h = self.fusion_proj(torch.cat([fused, cross], dim=-1))
        h = self.fusion_drop(h)
        return h

    def forward(self, z_eeg, z_audio, mask_eeg=None, mask_audio=None, return_window=False):
        e = self.e_proj(z_eeg)
        a = self.a_proj(z_audio)

        e_pool = self._pool(e, mask_eeg)
        a_pool = self._pool(a, mask_audio)

        h = self._fuse(e_pool, a_pool)
        logits = self.head(h).squeeze(-1)

        if return_window:
            win_h = self._fuse(e, a)
            win_logits_post = self.head(win_h).squeeze(-1)
            return logits, None, win_logits_post
        return logits

    def forward_per_window(self, z_eeg, z_audio, mask_eeg=None, mask_audio=None):
        e = self.e_proj(z_eeg)
        a = self.a_proj(z_audio)
        win_h = self._fuse(e, a)
        win_logits = self.head(win_h).squeeze(-1)
        ec_mask = mask_eeg if mask_eeg is not None else mask_audio
        if ec_mask is not None:
            win_logits = win_logits.masked_fill(ec_mask == 0, float("-inf"))
        return win_logits
