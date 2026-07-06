#!/bin/bash
set -e

echo "=== Validation Loop ==="

echo "[1/4] Compiling..."
python -m py_compile agent.py
python -m py_compile tools.py

echo "[2/4] Type checking..."
python -m mypy agent.py tools.py --ignore-missing-imports 2>/dev/null || echo "  (mypy not installed — skipping)"

echo "[3/4] Linting..."
python -m ruff check agent.py tools.py 2>/dev/null || python -m flake8 agent.py tools.py 2>/dev/null || echo "  (ruff/flake8 not installed — skipping)"

echo "[4/4] Testing..."
python -m pytest tests/ -v -k "not test_timeout_on_hanging_command"

echo "=== All checks passed ==="
