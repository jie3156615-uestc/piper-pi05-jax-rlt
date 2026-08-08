# Piper + OpenPI/JAX 实机推理运行说明

## 重要警告

本文档中的命令会使机械臂和夹爪真实运动。

- 必须有人在现场全程观察；
- 必须确保工作区无人、无障碍物；
- 必须保证物理急停随时可按；
- 不得使用 `nohup`、`&`、后台任务或无人值守方式运行；
- `Ctrl+C` 是软件停止并保持当前位置，不能代替物理急停；
- 停止 policy server 不能代替机械臂急停。

当前控制器已经完成一次 10 秒实机验证。该次验证执行 35 次重规划、175 个控制步，最大跟踪误差 0.232°，JAX 推理 P95 为 108 ms，机械臂错误码为 0。

## 一、控制参数

- 控制频率：30 Hz；
- 每次模型推理后执行前 5 个 action，然后重新采集图像和 CAN 状态；
- Piper 模式：MOVE_J；
- Piper 速度比例：10%；
- 软件最大关节速度：`0.03 rad/s`；
- 软件最大关节加速度：`0.2 rad/s²`；
- 每次重规划相对当前状态最多移动 8°；
- 夹爪范围：0–80 mm；
- 夹爪单步最大变化：2 mm；
- 单次运行时长必须在 0–600 秒之间。

官方关节限位：

| 关节 | 最小角度 | 最大角度 |
|---|---:|---:|
| J1 | -150° | 150° |
| J2 | 0° | 180° |
| J3 | -170° | 0° |
| J4 | -100° | 100° |
| J5 | -70° | 70° |
| J6 | -180° | 180° |

模型目标轻微越界时会被裁剪；超过官方限位 5°以上会立即停止并保持当前位置。

## 二、每次运行前必须检查

### 1. 登录并进入目录

```bash
ssh cwzk@192.168.2.26
cd ~/piper_jax_inference_v1
```

### 2. 检查 CAN

```bash
ip -details -statistics link show can0
```

必须包含：

```text
can state ERROR-ACTIVE
bitrate 1000000
bus-errors 0
```

如果 `can0` 不存在：

```bash
cd ~/vendor/piper_sdk_official/piper_sdk
sudo bash can_activate.sh can0 1000000
cd ~/piper_jax_inference_v1
```

### 3. 检查相机

```bash
rs-enumerate-devices -s
lsusb -t
```

必须看到：

- D435 `347522072112`，作为 `camera1`；
- D405 `260622272544`，作为 `camera2`；
- D435 必须处于 `5000M` USB 3.x 链路。

### 4. 检查 policy server

```bash
systemctl --user start openpi-piper-policy.service
timeout 120 bash -c 'until ss -ltn | grep -q "127.0.0.1:8000"; do sleep 2; done'
systemctl --user status openpi-piper-policy.service --no-pager
```

必须为 `active (running)`，并监听 `127.0.0.1:8000`。

### 5. 检查机械臂只读反馈

```bash
cd ~/vendor/piper_sdk_official/piper_sdk/demo/V2
source ~/venvs/pika/bin/activate
timeout 2s python -u piper_read_all_fps.py
```

正常值约为：

```text
all_fps: 3040
status: 200
joint_states: 200
gripper_msg: 200
```

状态必须是 `CAN_CTRL / NORMAL`，错误码必须是 0。

返回运行目录：

```bash
cd ~/piper_jax_inference_v1
```

### 6. 检查场景

确认以下项目后才能运行：

1. 红杯和盒子位于训练相机视野内；
2. 两台相机位置、朝向和训练时一致；
3. 机械臂周围没有人员、线缆或其他障碍物；
4. 机械臂初始姿态适合本次任务；
5. 物理急停可立即操作。

## 三、10 秒实机测试命令

建议每次改变场景、模型、相机或代码后，先执行 10 秒：

```bash
cd ~/piper_jax_inference_v1
RUN_ID=$(date +%Y%m%d_%H%M%S)_10s

PYTHONPATH=. ~/venvs/pika/bin/python -u -m piper_runtime.policy_hardware_rollout \
  --duration 10 \
  --authorization I_UNDERSTAND_POLICY_MOVES_ARM \
  --audit "reports/${RUN_ID}_audit.jsonl" \
  --report "reports/${RUN_ID}_report.json"
```

看到以下结果才表示正常结束：

```text
"outcome": "duration_complete"
"error": null
```

## 四、60 秒模型效果测试命令

只有 10 秒运行正常后才执行：

```bash
cd ~/piper_jax_inference_v1
RUN_ID=$(date +%Y%m%d_%H%M%S)_60s

PYTHONPATH=. ~/venvs/pika/bin/python -u -m piper_runtime.policy_hardware_rollout \
  --duration 60 \
  --authorization I_UNDERSTAND_POLICY_MOVES_ARM \
  --audit "reports/${RUN_ID}_audit.jsonl" \
  --report "reports/${RUN_ID}_report.json"
```

操作员必须在终端前和机械臂现场持续观察。任务完成、动作异常或即将碰撞时立即按 `Ctrl+C`；情况紧急时直接使用物理急停。

## 五、600 秒长时间运行命令

只有多次 60 秒运行稳定后才使用。600 秒为程序允许的最大时长：

```bash
cd ~/piper_jax_inference_v1
RUN_ID=$(date +%Y%m%d_%H%M%S)_600s

PYTHONPATH=. ~/venvs/pika/bin/python -u -m piper_runtime.policy_hardware_rollout \
  --duration 600 \
  --authorization I_UNDERSTAND_POLICY_MOVES_ARM \
  --audit "reports/${RUN_ID}_audit.jsonl" \
  --report "reports/${RUN_ID}_report.json"
```

600 秒运行仍然必须有人值守。不要通过 SSH 后关闭终端，不要将命令放入 systemd、cron、tmux 自动任务或后台进程。

## 六、停止方法

### 正常软件停止

在运行终端按：

```text
Ctrl+C
```

程序会捕获 `SIGINT`，向机械臂发送约 1 秒的当前位置保持命令，然后关闭相机和 CAN 客户端。

程序也会捕获 `SIGTERM` 和 `SIGHUP`，用于 SSH 或进程终止时尽量进入相同的保持流程。

### 紧急停止

出现碰撞风险、人员进入工作区、机械臂失控或软件停止无响应时，立即使用物理急停。

不要依赖以下命令作为急停：

```bash
systemctl --user stop openpi-piper-policy.service
```

该命令只停止 GPU 推理服务，不能保证机械臂立即停止。

## 七、运行结果和日志

每次命令生成两个文件：

```text
reports/<RUN_ID>_audit.jsonl
reports/<RUN_ID>_report.json
```

`audit.jsonl` 每发送一步就实时写盘，记录：

- 推理编号和 action 序号；
- 本次 CAN 状态快照；
- 实时反馈；
- 模型原始绝对目标；
- 安全过滤后的真实发送目标；
- 限位、速度、加速度和夹爪裁剪原因；
- 推理延迟和运行时间。

查看最终报告：

```bash
cat "reports/${RUN_ID}_report.json"
```

查看审计行数：

```bash
wc -l "reports/${RUN_ID}_audit.jsonl"
```

查看 policy server 日志：

```bash
tail -f ~/piper_jax_inference_v1/logs/policy_server.log
```

## 八、运行结束后的状态

正常到时或 `Ctrl+C` 后，机械臂会保持最后位置，不会自动回零。下一次运行会从当前 CAN 反馈姿态继续规划。

不要在未清空工作区时直接运行官方回零 demo。需要回零时应作为单独的受控动作执行。

## 九、已验证的 10 秒实机结果

报告位置：

```text
~/piper_jax_inference_v1/reports/policy_hardware_10s_report.json
~/piper_jax_inference_v1/reports/policy_hardware_10s_audit.jsonl
```

结果：

- `outcome=duration_complete`；
- `error=null`；
- 35 次重规划；
- 175 个控制步；
- 最大跟踪误差 0.232°；
- 推理延迟 P50 92.7 ms、P95 108.0 ms、最大 112.9 ms；
- 机械臂结束后为 `NORMAL`；
- 错误码 0；
- CAN 无总线错误。
