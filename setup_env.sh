#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${1:-venv}"

if [[ ! -f requirements.lock.txt ]]; then
  echo "requirements.lock.txt not found in repo root."
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/pip" install -r requirements.lock.txt
"${VENV_DIR}/bin/pip" install -e .

echo "Environment ready: ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
