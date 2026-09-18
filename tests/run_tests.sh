#!/usr/bin/env bash
# GridWise Unified Test Runner
# Runs both unit/integration tests and live HTTP public sample tests.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================================="
echo "          GridWise Unified Test Runner"
echo "=========================================================="

# Select Python binary
if [ -f "$REPO_ROOT/gridwise-llm/venv/bin/python" ]; then
    PYTHON="$REPO_ROOT/gridwise-llm/venv/bin/python"
elif [ -f "$REPO_ROOT/venv/bin/python" ]; then
    PYTHON="$REPO_ROOT/venv/bin/python"
else
    PYTHON="python3"
fi

echo "[*] Using Python: $($PYTHON --version)"

# 1. In-Memory Unit & Integration Tests
echo ""
echo "[Step 1/2] Running Unit & Integration Test Suite (test_api.py)..."
$PYTHON "$SCRIPT_DIR/test_api.py"

# 2. If a local server is running on port 8000, test live public samples
echo ""
echo "[Step 2/2] Checking for live HTTP server on http://localhost:8000..."
if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "[+] Live server detected! Running public sample scenarios..."
    $PYTHON "$SCRIPT_DIR/test_public_samples.py" http://localhost:8000
else
    echo "[-] No server running on http://localhost:8000."
    echo "    To test against a live server, run: uvicorn app.main:app (in gridwise-llm)"
    echo "    Then run: $PYTHON $SCRIPT_DIR/test_public_samples.py http://localhost:8000"
fi

echo ""
echo "=========================================================="
echo "          ALL TEST CHECKS COMPLETED!"
echo "=========================================================="

