#!/usr/bin/env bash
# Regenerate everything TOUR.md shows: the three diagrams and the app screenshots.
#
#   scripts/tour/make_tour.sh            # from the repository root
#
# Needs Node.js, Google Chrome, and `npm install` once in scripts/tour/. Starts its
# own app server on a spare port and stops it afterwards. Commit first: the Splits
# tab shows a warning when the working tree is dirty, and it would be captured.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
PORT=${PORT:-8599}

"$PY" scripts/tour/make_figures.py
"$PY" -m streamlit run app.py --server.port "$PORT" --server.headless true >/dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
node scripts/tour/capture.mjs "$PORT" docs/tour
