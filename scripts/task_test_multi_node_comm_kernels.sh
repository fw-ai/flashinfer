#!/bin/bash

set -eo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Source test environment setup (handles package overrides like TVM-FFI)
source "${SCRIPT_DIR}/setup_test_env.sh"

# Clean Python bytecode cache to avoid stale imports (e.g., after module refactoring)
# echo "Cleaning Python bytecode cache..."
# find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
# find . -type f -name '*.pyc' -delete 2>/dev/null || true
# echo "Cache cleaned."
# echo ""

# pip install -e . -v

pytest -s tests/comm/test_mnnvl_memory.py
pytest -s tests/comm/test_trtllm_mnnvl_allreduce.py
pytest -s tests/comm/test_mnnvl_moe_alltoall.py
