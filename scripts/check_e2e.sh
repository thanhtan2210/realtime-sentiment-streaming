#!/usr/bin/env bash
# End-to-end check used by .github/workflows/e2e.yml (and runnable by hand).
#
#   bash scripts/check_e2e.sh [EXPECTED_PREDICTIONS] [TIMEOUT_SECONDS]
#
# Passes when, within the timeout, the `predictions` collection holds exactly
# EXPECTED documents and `airline_windows` holds at least one, and the
# prediction count is still exactly EXPECTED three triggers later.
set -euo pipefail

EXPECTED=${1:-500}
TIMEOUT=${2:-300}

count() {
  docker compose exec -T mongodb mongosh --quiet sentiment \
    --eval "print(db.$1.countDocuments({}))" | tr -d '\r' | tail -n 1
}

deadline=$((SECONDS + TIMEOUT))
while true; do
  predictions=$(count predictions)
  windows=$(count airline_windows)
  echo "t=${SECONDS}s predictions=${predictions} airline_windows=${windows}"
  if [ "$predictions" -gt "$EXPECTED" ]; then
    echo "FAIL: more than ${EXPECTED} predictions"
    exit 1
  fi
  if [ "$predictions" -eq "$EXPECTED" ] && [ "$windows" -ge 1 ]; then
    break
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "FAIL: timed out after ${TIMEOUT}s"
    exit 1
  fi
  sleep 5
done

sleep 15
predictions=$(count predictions)
if [ "$predictions" -ne "$EXPECTED" ]; then
  echo "FAIL: prediction count changed to ${predictions}"
  exit 1
fi
echo "OK: ${predictions} predictions, $(count airline_windows) airline windows"
