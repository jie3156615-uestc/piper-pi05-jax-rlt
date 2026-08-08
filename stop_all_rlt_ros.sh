#!/usr/bin/env bash
set -o pipefail

# Catkin setup files are not nounset-safe. Source first, then enable strict
# handling for this script; otherwise shutdown can silently exit before doing
# any work when inherited environment variables are absent.
set +u
source "$HOME/pika_ros/install/setup.bash" 2>/dev/null || true
set -u

readonly RLT_JOB_PATTERN='piper_runtime[.]rlt_online_session|piper_runtime[.]rlt_takeover_rollout|run_rlt_online_update_hook[^ ]*[.]sh|run_online_rlt_update[.]py|generate_external_rlt_enrichment_cache[.]py|prepare_external_rlt_replay[.]py|train_real_rlt_jax[.]py'
readonly RLT_ROS_PROCESS_PATTERN='start_sensor_gripper[.]bash|open_sensor_gripper[.]launch|open_single_sensor[.]launch|teleop_rand_single_piper[.]launch'
readonly RLT_ROS_MASTER_PATTERN='/opt/ros/noetic/bin/roslaunch|/opt/ros/noetic/bin/roscore|/opt/ros/noetic/bin/rosmaster'

readonly -a RLT_USER_SERVICES=(
  rlt-native-sdk-command-gripper-v3.service
  openpi-rlt-shadow-policy-gripper-v3.service
  rlt-native-sdk-command.service
  openpi-rlt-shadow-policy.service
  openpi-piper-policy.service
  rlt-piper-controller.service
  rlt-teleop-controller.service
  rlt-takeover-sources.service
  rlt-roscore.service
)

echo "[1/4] Stop online rollout and learner jobs"
pkill -INT -f "$RLT_JOB_PATTERN" 2>/dev/null || true
PURE_ROLLOUT_PIDS="$(ps -eo pid=,comm=,args= | awk '$2 ~ /^python/ && $0 ~ / -m piper_runtime[.]policy_hardware_rollout( |$)/ {print $1}')"
if [[ -n "$PURE_ROLLOUT_PIDS" ]]; then
  kill -INT $PURE_ROLLOUT_PIDS 2>/dev/null || true
fi
sleep 2
pkill -TERM -f "$RLT_JOB_PATTERN" 2>/dev/null || true
PURE_ROLLOUT_PIDS="$(ps -eo pid=,comm=,args= | awk '$2 ~ /^python/ && $0 ~ / -m piper_runtime[.]policy_hardware_rollout( |$)/ {print $1}')"
if [[ -n "$PURE_ROLLOUT_PIDS" ]]; then
  kill -TERM $PURE_ROLLOUT_PIDS 2>/dev/null || true
fi

echo "[2/4] Disable Piper when the ROS service is still available"
if timeout 2 rosservice list 2>/dev/null | grep -qx /enable_srv; then
  timeout 5 rosservice call /enable_srv "{enable_request: false}" >/dev/null 2>&1 || true
fi

echo "[3/4] Stop RLT/Pika/policy services and ROS nodes"
systemctl --user stop "${RLT_USER_SERVICES[@]}" 2>/dev/null || true
timeout 5 rosnode kill -a >/dev/null 2>&1 || true
# Also terminate legacy/manual launch chains.  In particular,
# start_sensor_gripper.bash restarts roslaunch in a loop unless its parent shell
# is stopped, which can otherwise recreate a ROS master after shutdown.
pkill -INT -f "$RLT_ROS_PROCESS_PATTERN" 2>/dev/null || true
pkill -INT -f "$RLT_ROS_MASTER_PATTERN" 2>/dev/null || true
sleep 2
pkill -TERM -f "$RLT_ROS_PROCESS_PATTERN" 2>/dev/null || true
pkill -TERM -f "$RLT_ROS_MASTER_PATTERN" 2>/dev/null || true
sleep 1
pkill -KILL -f "$RLT_ROS_PROCESS_PATTERN" 2>/dev/null || true
pkill -KILL -f "$RLT_ROS_MASTER_PATTERN" 2>/dev/null || true
# roslaunch can be SIGSTOPed or killed before it reaps children. Remove only
# the known Piper/Pika ROS executables so no orphan keeps USB or CAN ownership.
pkill -KILL -f '/pika_locator/pika_single_locator_node|/sensor_tools/serial_gripper_imu|/piper/scripts/piper_ctrl_single_node[.]py|/pika_remote_piper/scripts/(piper_FK|piper_IK|teleop_piper_publish)[.]py' 2>/dev/null || true

echo "[4/4] Verify"
PURE_ROLLOUT_PIDS="$(ps -eo pid=,comm=,args= | awk '$2 ~ /^python/ && $0 ~ / -m piper_runtime[.]policy_hardware_rollout( |$)/ {print $1}')"
ACTIVE_SERVICES=()
for unit in "${RLT_USER_SERVICES[@]}"; do
  if systemctl --user --quiet is-active "$unit"; then
    ACTIVE_SERVICES+=("$unit")
  fi
done
if ((${#ACTIVE_SERVICES[@]})) \
  || pgrep -f "$RLT_JOB_PATTERN|/opt/ros/noetic/bin/rosmaster" >/dev/null \
  || [[ -n "$PURE_ROLLOUT_PIDS" ]]; then
  echo "Some RLT/ROS components are still active; inspect with ps/systemctl." >&2
  if ((${#ACTIVE_SERVICES[@]})); then
    printf '  active service: %s\n' "${ACTIVE_SERVICES[@]}" >&2
  fi
  pgrep -af "$RLT_JOB_PATTERN|/opt/ros/noetic/bin/rosmaster" >&2 || true
  exit 1
fi

echo "ONE_CLICK_STOP_PASS: JAX inference, gripper-v3/legacy RLT, learner, Pika/ROS and policy services are stopped."
echo "Piper disable was requested before ROS shutdown; can0 is intentionally left UP."
