#!/usr/bin/env bash
set -eo pipefail

ROOT="$HOME/piper_jax_inference_v1"
SENSE_SERIAL_PATH="/dev/serial/by-path/pci-0000:00:14.0-usb-0:1.4:1.0-port0"

if [[ ! -e "$SENSE_SERIAL_PATH" ]]; then
  echo "Sense demonstrator serial device is missing: $SENSE_SERIAL_PATH" >&2
  echo "Reconnect the Sense USB device on physical port 1-1.4, then retry." >&2
  exit 1
fi
if [[ -e /dev/ttyUSB60 ]] \
   && [[ "$(readlink -f "$SENSE_SERIAL_PATH")" == "$(readlink -f /dev/ttyUSB60)" ]]; then
  echo "Sense and robot gripper resolve to the same serial device; refusing unsafe RLT startup." >&2
  exit 1
fi

if pgrep -af '[/]opt/ros/noetic/bin/roslaunch .*run_data_capture' >/dev/null \
   || pgrep -af '[/]opt/ros/noetic/bin/roslaunch .*open_sensor_gripper.launch' >/dev/null; then
  echo "RLT cannot start while original Pika camera/data-capture mode owns the RealSense devices." >&2
  echo "Stop start_sensor_gripper/open_sensor_gripper first, then retry." >&2
  exit 1
fi

pure_rollout_pids() {
  ps -eo pid=,comm=,args= \
    | awk '$2 ~ /^python/ && $0 ~ / -m piper_runtime[.]policy_hardware_rollout( |$)/ {print $1}'
}

if [[ -n "$(pure_rollout_pids)" ]]; then
  echo "A pure policy rollout is still running or suspended. Stop it before starting RLT." >&2
  ps -o pid,ppid,stat,cmd -p "$(pure_rollout_pids | paste -sd, -)" >&2 || true
  exit 1
fi

# Pure inference and RLT both use a node named /piper_ctrl_single_node.  They
# must never be active together: duplicate nodes make reset commands appear to
# publish successfully while the actual CAN controller is being replaced.
systemctl --user stop rlt-piper-controller.service 2>/dev/null || true
systemctl --user stop openpi-piper-policy.service 2>/dev/null || true
systemctl --user stop rlt-native-sdk-command.service 2>/dev/null || true

systemctl --user start rlt-roscore.service
source /opt/ros/noetic/setup.bash
source "$HOME/pika_ros/install/setup.bash"
set -u
for _ in $(seq 1 50); do
  ROS_MASTER_URI=http://localhost:11311 rosparam list >/dev/null 2>&1 && break
  sleep 0.2
done

systemctl --user restart rlt-takeover-sources.service
systemctl --user restart rlt-teleop-controller.service

stable=0
for _ in $(seq 1 100); do
  controller_count="$(pgrep -fc '[p]iper_ctrl_single_node.py' || true)"
  sense_device="$(rosparam get /sensor_serial_gripper_imu/serial_port 2>/dev/null || true)"
  if systemctl --user is-active --quiet rlt-teleop-controller.service \
     && ! systemctl --user is-active --quiet rlt-piper-controller.service \
     && rosservice list 2>/dev/null | grep -qx '/enable_srv' \
     && rostopic list 2>/dev/null | grep -qx '/joint_states_single' \
     && rosnode list 2>/dev/null | grep -qx '/sensor_serial_gripper_imu' \
     && [[ "$sense_device" == "$(readlink -f "$SENSE_SERIAL_PATH")" ]] \
     && [[ "$controller_count" == "1" ]]; then
    stable=$((stable + 1))
    if [[ "$stable" -ge 5 ]]; then
      echo "RLT infrastructure ready."
      echo "  ROS master             : ready"
      echo "  Piper controller count : 1 (RLT controller only)"
      echo "  Pika/Sense sources     : started"
      echo "  Sense serial           : $sense_device (physical USB 1-1.4, verified by ROS parameter)"
      echo "  RealSense ROS nodes    : not started; cameras are free for RLT"
      exit 0
    fi
  else
    stable=0
  fi
  sleep 0.2
done

echo "RLT infrastructure failed to become ready or more than one Piper controller exists." >&2
systemctl --user status rlt-roscore.service rlt-takeover-sources.service rlt-teleop-controller.service --no-pager || true
pgrep -af '[p]iper_ctrl_single_node.py' || true
exit 1
