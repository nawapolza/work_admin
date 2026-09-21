#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
npm --prefix frontend install
[ -f .env ] || cp .env.example .env
echo "Setup complete. Put models in backend/models then run: bash run_linux.sh"

