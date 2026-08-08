#!/usr/bin/env bash
set -euo pipefail

source "$HOME/pika_ros/install/setup.bash"

echo "--- nodes ---"
rosnode list 2>/dev/null | sort || true

echo "--- topic info ---"
for topic in /pika_pose /joint_states /joint_states_gripper /piper_IK/ctrl_end_pose /joint_states_single; do
  echo
  echo "### ${topic}"
  rostopic info "${topic}" 2>/dev/null || true
done

echo
echo "--- short hz checks ---"
for topic in /pika_pose /joint_states /joint_states_gripper /joint_states_single; do
  echo
  echo "### ${topic}"
  timeout 3s rostopic hz "${topic}" 2>/dev/null || true
done

echo
echo "--- status samples ---"
for topic in /pika_localization_status /teleop_status /arm_status; do
  echo
  echo "### ${topic}"
  timeout 3s rostopic echo -n 1 "${topic}" 2>/dev/null || true
done

echo
echo "--- hidraw permissions ---"
stat -c '%A %U %G %n' /dev/hidraw* 2>/dev/null || true
