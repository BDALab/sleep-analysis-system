# macOS setup

The project data can remain on the external ExFAT drive, but the Python virtual
environment must be stored on the Mac's internal APFS disk. ExFAT creates
AppleDouble `._*` files inside `site-packages`, which can break Matplotlib,
SHAP, and pip package discovery.

## Recommended setup

```bash
cd /Volumes/Portable/geneactiv-processing-data
bash scripts/setup_macos.sh
source ~/.virtualenvs/geneactiv-processing-data-py313/bin/activate
python manage.py runserver
```

The setup script installs Homebrew `libomp`, creates a Python 3.13 environment
under `~/.virtualenvs`, installs `requirements.txt`, tests scientific imports,
runs a small XGBoost fit, and runs Django's system check.

## XGBoost runtime

Apple Silicon uses CPU histogram training because XGBoost CUDA acceleration
requires an NVIDIA CUDA GPU and is not available on macOS.

Defaults:

```bash
GENEACTIV_XGB_DEVICE=cpu
GENEACTIV_XGB_TREE_METHOD=hist
GENEACTIV_XGB_THREADS=8
```

The variables are optional. To reduce heat or memory pressure, for example:

```bash
export GENEACTIV_XGB_THREADS=4
```

On a CUDA-capable Linux or Windows machine, `GENEACTIV_XGB_DEVICE=cuda` can be
used explicitly. With no override, the project selects CPU on macOS and detects
NVIDIA CUDA on other platforms.
