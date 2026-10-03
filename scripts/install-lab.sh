#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3,10), "Требуется Python 3.10+; системный Python не заменяйте."'
"$PYTHON" -m venv .venv
if [[ "${1:-}" == "--offline" ]]; then
  : "${WHEELHOUSE:?Для офлайн-установки задайте WHEELHOUSE с совместимыми колёсами}"
  .venv/bin/python -m pip install --no-index --find-links "$WHEELHOUSE" 'setuptools>=68' wheel
  .venv/bin/python -m pip install --no-index --find-links "$WHEELHOUSE" --no-build-isolation -e '.[test,data,lab]'
else
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/python -m pip install -e '.[test,data,lab]'
fi
printf '\nЗапуск: .venv/bin/python -m global_weather.lab.app --workspace outputs/lab\n'
printf 'Интерфейс: http://127.0.0.1:8765\n'
