#!/bin/bash
set -e

echo "=== Validation Loop ==="

echo "[1/4] Linting..."
if command -v ruff &> /dev/null; then
    ruff check .
else
    echo "  ruff not found — skipping lint"
fi

echo "[2/4] Type checking..."
if command -v mypy &> /dev/null; then
    mypy agent.py tools.py tests/
else
    echo "  mypy not found — skipping type check"
fi

echo "[3/4] Syntax check..."
python -m py_compile agent.py
python -m py_compile tools.py

echo "[4/4] Testing..."
python -m pytest tests/ -v --tb=short

echo "=== All checks passed ==="
