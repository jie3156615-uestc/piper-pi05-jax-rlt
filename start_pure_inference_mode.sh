#!/usr/bin/env bash
set -eo pipefail

ROOT="$HOME/piper_jax_inference_v1"

if pgrep -af '[/]opt/ros/noetic/bin/roslaunch .*run_data_capture' >/dev/null \
   || pgrep -af '[/]opt/ros/noetic/bin/roslaunch .*open_sensor_gripper.launch' >/dev/null; then
  echo "Pure inference cannot start while original Pika camera/data-capture mode owns the RealSense devices."
  echo "Stop start_sensor_gripper/open_sensor_gripper first, then retry. Existing datasets are unaffected."
  exit 1
fi

# A suspended rollout still owns Python/ROS resources even though it appears idle.
if pgrep -af '[p]iper_runtime.policy_hardware_rollout' >/dev/null; then
  echo "A policy_hardware_rollout process is already running or suspended. Terminate it before starting a new run."
  exit 1
fi

systemctl --user stop rlt-teleop-controller.service rlt-takeover-sources.service 2>/dev/null || true
systemctl --user start rlt-roscore.service

source /opt/ros/noetic/setup.bash
set -u
for _ in $(seq 1 50); do
  ROS_MASTER_URI=http://localhost:11311 rosparam list >/dev/null 2>&1 && break
  sleep 0.2
done

systemctl --user restart rlt-piper-controller.service
systemctl --user start openpi-piper-policy.service

for _ in $(seq 1 100); do
  if rosservice list 2>/dev/null | grep -qx '/enable_srv' \
     && rostopic list 2>/dev/null | grep -qx '/joint_states_single' \
     && ss -ltn 2>/dev/null | grep -q '127.0.0.1:8000'; then
    echo "Pure JAX inference mode ready."
    echo "  ROS/Piper controller: ready"
    echo "  Policy server 8000  : ready"
    echo "  Pika teleop         : not started (not required)"
    echo "  RealSense ROS nodes : not started; cameras are free for policy_hardware_rollout"
    exit 0
  fi
  sleep 0.2
done

echo "Pure inference mode failed to become ready."
systemctl --user status rlt-piper-controller.service openpi-piper-policy.service --no-pager || true
exit 1
