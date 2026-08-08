#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/piper_jax_inference_v1"
EPISODES="${EPISODES:-1}"
RESET_SECONDS="${RESET_SECONDS:-4}"
H50_PREFETCH_LEAD_STEPS="${H50_PREFETCH_LEAD_STEPS:-4}"
RUN_ID="h50_$(date +%Y%m%d_%H%M%S)"
LABELS_CSV="reports/${RUN_ID}_labels.csv"

cd "$ROOT"
mkdir -p reports
printf 'episode,success,report_path,audit_path,recorded_at\n' > "$LABELS_CSV"

echo "H50 pure inference"
echo "  episode mode    : ${EPISODES} episode(s); each episode starts a fresh rollout process"
echo "  mandatory start: ${RESET_SECONDS}s smooth reset to the unchanged shared 7-D SFT start"
echo "  stop condition  : operator label 1/0 during rollout; no duration or plan-count limit"
echo "  execute steps   : 50"
echo "  H50 prefetch    : ${H50_PREFETCH_LEAD_STEPS} steps before boundary (standby + atomic handoff)"
echo "  handoff         : drop stale prefix; 5-step velocity-aware local bridge; never globally rebase"
echo "  accuracy guard  : reject only if local bridge excursion exceeds 0.12rad"
echo "  bridge bound    : velocity-target correction is hard-clipped to 0.12rad per joint"
echo "  boundary hold   : feedback-anchored brake/hold when command tracking error exceeds 0.03rad"
echo "  audit           : $ROOT/reports/${RUN_ID}_audit.jsonl"
echo "  report          : $ROOT/reports/${RUN_ID}_report.json"
echo "  labels          : $ROOT/${LABELS_CSV}"
echo "  stop            : Ctrl-C"

if ! [[ "$EPISODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPISODES must be a positive integer: $EPISODES" >&2
  exit 2
fi

for episode in $(seq 1 "$EPISODES"); do
  EP_ID="$(printf '%03d' "$episode")"
  AUDIT_PATH="reports/${RUN_ID}_ep${EP_ID}_audit.jsonl"
  REPORT_PATH="reports/${RUN_ID}_ep${EP_ID}_report.json"
  echo
  echo "[h50] episode ${episode}/${EPISODES}: reset -> policy inference -> execute"
  PYTHONPATH=. "${HOME}/venvs/pika/bin/python" -u -m piper_runtime.policy_hardware_rollout \
    --execute-steps 50 \
    --operator-label-control \
    --reset-seconds "$RESET_SECONDS" \
    --h50-prefetch-lead-steps "$H50_PREFETCH_LEAD_STEPS" \
    --safety-profile native \
    --hardware-io ros_controller \
    --authorization I_UNDERSTAND_POLICY_MOVES_ARM \
    --audit "$AUDIT_PATH" \
    --report "$REPORT_PATH"

  EPISODE_SUCCESS="$("${HOME}/venvs/pika/bin/python" - "$REPORT_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    report = json.load(f)
label = report.get("operator_label")
if label in {"1", "0"}:
    print(label)
PY
)"
  if [[ "$EPISODE_SUCCESS" != "1" && "$EPISODE_SUCCESS" != "0" ]]; then
    echo "[h50] no 1/0 label found in $REPORT_PATH; stopping."
    exit 0
  fi
  printf '%s,%s,%s,%s,%s\n' \
    "$episode" "$EPISODE_SUCCESS" "$REPORT_PATH" "$AUDIT_PATH" "$(date -Is)" \
    >> "$LABELS_CSV"
  echo "[h50] recorded success=${EPISODE_SUCCESS}; labels: $ROOT/${LABELS_CSV}"

  if (( episode < EPISODES )); then
    echo "[h50] starting next episode automatically."
  fi
done
