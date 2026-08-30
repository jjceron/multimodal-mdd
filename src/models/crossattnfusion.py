from __future__ import annotations

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


class CrossModalFusion(nn.Module):
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
        e_out, _ = self._attend(self.e_q(eeg), self.a_k(audio), self.a_v(audio), self.e_o, mask_audio)
        eeg = self.norm_e(eeg + e_out)
        a_out, _ = self._attend(self.a_q(audio), self.e_k(eeg), self.e_v(eeg), self.a_o, mask_eeg)
        audio = self.norm_a(audio + a_out)
        return eeg, audio


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=mask)[0]
        return x + self.mlp(self.norm2(x))


class CrossAttnFusion(nn.Module):
    def __init__(
        self,
        eeg_dim,
        aud_dim,
        hidden=32,
        n_heads=2,
        n_self_attn_layers=1,
        dropout=0.5,
        attn_dropout=0.1,
    ):
        super().__init__()
        self.eeg_proj = nn.Linear(eeg_dim, hidden)
        self.aud_proj = nn.Linear(aud_dim, hidden)
        self.eeg_rms = RMSNorm(hidden)
        self.aud_rms = RMSNorm(hidden)
        self.feat_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.cross_layers = nn.ModuleList(
            [CrossModalFusion(hidden, n_heads, attn_dropout=attn_dropout)]
        )
        self.self_attn_layers = nn.ModuleList(
            [SelfAttentionBlock(hidden, n_heads, attn_dropout) for _ in range(n_self_attn_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, z_eeg, z_audio, mask_eeg=None):
        e = self.eeg_rms(self.eeg_proj(z_eeg))
        a = self.aud_rms(self.aud_proj(z_audio))
        if self.training:
            e = self.feat_drop(e)
            a = self.feat_drop(a)
        for layer in self.cross_layers:
            e, a = layer(e, a, mask_eeg, None)
        z = 0.5 * (e + a)
        attn_mask = (mask_eeg == 0) if mask_eeg is not None else None
        for layer in self.self_attn_layers:
            z = layer(z, mask=attn_mask)
        if mask_eeg is not None:
            n = mask_eeg.sum(dim=1, keepdim=True).clamp(min=1)
            z = (z * mask_eeg.unsqueeze(-1)).sum(dim=1) / n
        else:
            z = z.mean(dim=1)
        return self.head(z).squeeze(-1)
