#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
trap 'kill 0' EXIT
python backend/app.py &
npm --prefix frontend run dev

