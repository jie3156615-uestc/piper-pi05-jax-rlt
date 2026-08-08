#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_RLT_REPO:-$HOME/piper_jax_inference_v1}"
PROFILE="${1:-offline}"

if [[ "$PROFILE" != "offline" && "$PROFILE" != "rlt" && "$PROFILE" != "inference" ]]; then
  echo "usage: $0 [offline|rlt|inference]" >&2
  exit 2
fi

cd "$ROOT"
python3 -m py_compile scripts/ops/piper_rlt_healthcheck.py
bash -n scripts/ops/run_ops_acceptance.sh
bash -n stop_all_rlt_ros.sh

PROJECT_PYTHON="${PIPER_RLT_PROJECT_PYTHON:-$ROOT/.venv/bin/python}"
if [[ -x "$PROJECT_PYTHON" ]]; then
  "$PROJECT_PYTHON" -m pytest -q \
    tests/test_ops_healthcheck.py \
    tests/test_policy_hardware_rollout.py \
    tests/test_rlt_takeover_rollout.py
else
  python3 -m pytest -q tests/test_ops_healthcheck.py
fi

if command -v systemd-analyze >/dev/null 2>&1; then
  systemd-analyze --user verify deploy/systemd/portable/*.service deploy/systemd/portable/*.timer
fi

if command -v logrotate >/dev/null 2>&1 && [[ -f "$HOME/.config/piper-rlt/logrotate.conf" ]]; then
  logrotate --debug --state "$HOME/.local/state/piper-rlt/logrotate.state" "$HOME/.config/piper-rlt/logrotate.conf"
fi

python3 scripts/ops/piper_rlt_healthcheck.py \
  --profile "$PROFILE" \
  --output "$HOME/.local/state/piper-rlt/health-latest.json"

cat <<'EOF'
Software acceptance passed. This script issued no robot commands.
Before unattended operation, complete the witnessed physical stop drill in docs/OPERATIONS.md.
EOF
