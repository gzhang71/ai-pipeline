#!/usr/bin/env bash
# Run every example end to end. Offline: no API key, no network.
#
#     bash examples/run_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"

for script in examples/0*.py; do
    echo
    echo "########################################################################"
    echo "# $script"
    echo "########################################################################"
    "$PY" "$script"
done

echo
echo "all examples completed"
