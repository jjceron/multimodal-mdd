### Create environment from scratch

```bash
conda create -n venv python=3.11
conda activate venv
conda install -C conda-forge poetry
conda env export --from-history > environment.yml
```


### Create environment from file

```bash
conda env create -f environment.yml
```



