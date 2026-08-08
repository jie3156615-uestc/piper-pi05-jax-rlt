# Piper / Pika / LeRobot 数采

参考主机使用 Piper 原生 ROS 状态与命令接口、Pika/Sense 示教和两台 RealSense。数采本身不由 policy server 驱动。

## 启动顺序

终端 1：

```bash
cd ~/pika_ros/src/PikaAnyArm/piper/piper_ros
bash can_activate.sh can0 1000000
roscore
```

终端 2：按 Pika ROS 的安装说明启动 sensor gripper。数采时由 Pika/Sense 接管，禁止同时启动纯推理或 RLT controller。

终端 3：启动 Piper teleop。具体 launch 名随 Pika ROS 版本而异；5090 的已验证覆盖文件在 `pika_overrides/`，ROS launch 文件在 `ros/`。

终端 4：检查相机并录制：

```bash
lerobot-find-cameras realsense
bash data_collection/record_lerobot_piper.sh
```

## 必查项目

- `camera1` 必须是全局 D435，`camera2` 必须是腕部 D405。
- 两路都应为 RGB 640×480@30，LeRobot dataset fps 为 30。
- 机械臂为 `record_only=true`，动作来自示教器/ROS；记录脚本不应自己驱动机械臂。
- 检查 action 与 state 的 30 ms 对齐补偿是否仍适合当前主机负载。
- 每个 episode 保存后抽查首帧、抓取、放置、终止帧；场景未复位或相机异常的数据不要进入 SFT/RLT。
