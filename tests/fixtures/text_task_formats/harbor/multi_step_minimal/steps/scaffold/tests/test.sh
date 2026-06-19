#!/usr/bin/env bash
set -euo pipefail
test "$(cat /app/step.txt 2>/dev/null || true)" = "done"
mkdir -p /logs/verifier
echo 1 > /logs/verifier/reward.txt
