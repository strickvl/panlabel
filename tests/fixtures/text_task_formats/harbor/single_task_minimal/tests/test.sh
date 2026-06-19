#!/usr/bin/env bash
set -euo pipefail
if [ "$(cat /app/answer.txt 2>/dev/null || true)" = "ok" ]; then
  mkdir -p /logs/verifier
  echo 1 > /logs/verifier/reward.txt
else
  mkdir -p /logs/verifier
  echo 0 > /logs/verifier/reward.txt
  exit 1
fi
