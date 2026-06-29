#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script is intended for macOS." >&2
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required: https://brew.sh" >&2
  exit 1
fi

if ! brew list --versions libomp >/dev/null 2>&1; then
  brew install libomp
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in python3.13 python3.12 python3.11; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="$(command -v "${candidate}")"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Python 3.11-3.13 is required. Install it with: brew install python@3.13" >&2
  exit 1
fi

PROJECT_NAME="$(basename "$(pwd)")"
# Keep the environment on APFS. ExFAT creates AppleDouble files inside
# site-packages, which can break Matplotlib, SHAP, and pip package discovery.
VENV_PATH="${VENV_PATH:-${HOME}/.virtualenvs/${PROJECT_NAME}-py313}"
mkdir -p "$(dirname "${VENV_PATH}")"
"${PYTHON_BIN}" -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_PATH}/bin/python" -m pip install -r requirements.txt

# Also protect explicitly configured environments that live on removable media.
find "${VENV_PATH}" -name '._*' -delete

GENEACTIV_XGB_DEVICE=cpu "${VENV_PATH}/bin/python" - <<'PY'
import numpy as np
import matplotlib
import pandas
import shap
import tables
import xgboost as xgb

X = np.array([[0.0], [1.0], [0.1], [0.9]], dtype=np.float32)
y = np.array([0, 1, 0, 1], dtype=np.int32)
model = xgb.XGBClassifier(
    tree_method="hist",
    device="cpu",
    n_jobs=2,
    n_estimators=2,
    max_depth=2,
)
model.fit(X, y)
print("macOS environment ready; scientific imports and XGBoost CPU smoke test passed.")
PY

"${VENV_PATH}/bin/python" manage.py check

echo "Activate with: source ${VENV_PATH}/bin/activate"
