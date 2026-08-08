#!/usr/bin/env bash
set -euo pipefail

CAN_PORT="${PIPER_CAN_INTERFACE:-can0}"
GLOBAL_SERIAL="${PIPER_GLOBAL_CAMERA_SERIAL:-347522072112}"
WRIST_SERIAL="${PIPER_WRIST_CAMERA_SERIAL:-260622272544}"
REPO_ID="${LEROBOT_REPO_ID:-local/piper_greenblock_5090_v2}"
ROOT="${LEROBOT_DATA_ROOT:-$HOME/lerobot_datasets/piper_greenblock_5090_v2}"
TASK="${LEROBOT_TASK:-Put the green block into the box.}"
NUM_EPISODES="${LEROBOT_NUM_EPISODES:-50}"
EPISODE_TIME="${LEROBOT_EPISODE_TIME_S:-30}"
RESET_TIME="${LEROBOT_RESET_TIME_S:-10}"

exec lerobot-record \
  --robot.type=piper \
  --robot.port="$CAN_PORT" \
  --robot.record_only=true \
  --robot.action_from_state_delay_s=0.03 \
  --robot.log_skipped_actions=false \
  --robot.cameras="{ camera1: {type: realsense, serial_number_or_name: $GLOBAL_SERIAL, width: 640, height: 480, fps: 30}, camera2: {type: realsense, serial_number_or_name: $WRIST_SERIAL, width: 640, height: 480, fps: 30} }" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$ROOT" \
  --dataset.fps=30 \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s="$EPISODE_TIME" \
  --dataset.reset_time_s="$RESET_TIME" \
  --dataset.single_task="$TASK" \
  --dataset.push_to_hub=false
