# Piper + π0.5 JAX + Online RLT

这是单臂 AgileX Piper 在真实硬件上运行 π0.5、JAX SFT 和在线 RLT 的可审计代码快照。仓库整理自 5090 主机上已成功运行的数采、SFT、纯推理、RL Token、ResNet 阶段门控和 Actor–Critic 在线更新链路；快照日期为 **2026-08-08**。

> **安全提示**：仓库会向真实机械臂发指令。第一次部署只做只读检查和 dry-run；确认急停、CAN、相机映射、关节方向、工作空间和现场人员均安全后，才能使用带 `I_UNDERSTAND_POLICY_MOVES_ARM` 的执行入口。验证指标不是物理安全证明。

## 仓库内容

| 模块 | 主要入口 | 说明 |
| --- | --- | --- |
| LeRobot 数采 | `data_collection/record_lerobot_piper.sh` | Piper 原生状态/动作、Pika/Sense 人工示教、双 RealSense、30 Hz |
| π0.5 SFT | `scripts/train.py`、`src/openpi/training/config.py` | 50 步动作块；前 6 维关节为相对首帧的 delta，夹爪保持绝对量 |
| 纯推理 | `start_jax_inference_mode.sh`、`run_h50_inference.sh` | policy server 常驻；H50 执行；无需 Sense；4 秒柔和复位 |
| 固定轮次评估 | `run_fixed_checkpoint_eval_gripper_close_v3.sh` | 固定 Actor checkpoint 的多 episode 真机评估 |
| 原始 RL Token | `scripts/train_rlt_token_jax.py`、`src/openpi/rlt/jax_token.py` | 冻结 π0.5 前缀 token，训练 JAX/Flax autoencoder，再导出 encoder-only |
| ResNet 阶段分类器 | `scripts/build_piper_phase_dataset_from_intervals.py`、`scripts/train_piper_phase_classifier.py` | 判断何时进入精细夹取/放置阶段，不替换 VLA 视觉 token |
| 在线 RLT | `start_jax_rlt_mode.sh`、`run_rlt_lineage_gripper_close_v3_online.sh` | H50 base + C10 残差 Actor、双 Q Critic、人工奖励/准入、严格 replay 合约 |
| 缓存和审计 | `scripts/piper_rlt/tools/` | 增量 RL-token cache、精确执行时 `a_ref`、同计划 C10、`t+10`、候选验证/回滚 |

模型权重、真实 episode、图像、视频、replay、RL-token cache、训练日志和在线 session **不在 Git 中**。它们体积大且可能包含现场信息；仓库仅提交代码、配置模板、指标和 SHA-256 指纹。完整放置约定见 `docs/ARTIFACTS.md`。

## 参考硬件与软件

这不是最低配置，而是已验证的 5090 主机快照。

| 项目 | 已验证配置 |
| --- | --- |
| 主机 | `cwzk-MS-7D99`，Ubuntu 20.04.6 LTS，Linux 5.15.0-139，x86_64 |
| CPU / 内存 | Intel Core i7-12700F，31 GiB RAM，2 GiB swap |
| GPU | NVIDIA GeForce RTX 5090 D v2，24,455 MiB，driver 580.126.09，CUDA compute capability 12.0 |
| 机械臂 | 单臂 AgileX Piper，CAN `can0`，1,000,000 bit/s |
| 示教器 | Pika/Sense；仅数采与 RLT 人工介入使用，纯推理不使用 |
| 全局相机 | Intel RealSense D435，序列号 `347522072112`，运行时键 `camera1` |
| 腕部相机 | Intel RealSense D405，序列号 `260622272544`，运行时键 `camera2` |
| ROS | ROS 1 Noetic |
| JAX 环境 | Python 3.11.15；JAX/JAXlib 0.5.3，Flax 0.10.2，Optax 0.2.8，Orbax 0.11.13 |
| 硬件/ROS 环境 | Python 3.8.10；piper-sdk 0.6.1，python-can 4.5.0，pyrealsense2 2.55.1.6486 |

已验证的上游版本：

- `agilexrobotics/piper_sdk`：`c05c5454b1cf61c05ad26385e0c0a3aa6d3c7bad`
- `agilexrobotics/pika_ros`：`f40dc2868d87a6285d60524c37733101aa5785ee`
- 5090 上检查到的 LeRobot checkout：`a5b29d430105f5235eb05bbf2db5a0d747a869d6`
- 本仓库 `uv.lock`/`pyproject.toml` 的 LeRobot 依赖固定为 `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`

最后两项是“现场 checkout”和“OpenPI 可复现依赖”两个不同概念，不应混用。

## 算法链路

1. π0.5 SFT 根据双相机、7 维状态和文本任务生成 `H=50` 的绝对执行参考动作。数据训练时将前 6 个关节维从绝对角转为相对动作块首状态的 delta；第 7 维夹爪保持绝对量。
2. 冻结的 π0.5 prefix embeddings 输入 RL Token autoencoder。在线只加载 encoder，得到 2,048 维 `z_rl`；它是 Actor/Critic 的条件特征，不取代 π0.5 的视觉 token。
3. ResNet-18 读取 `camera1|camera2` 横向拼接图，连续 3 帧概率不低于 0.5 后单次锁存，直到 episode 终止；RLT 只在这个精细阶段介入。
4. Actor 输出 10 步 C10 残差，叠加到同一 H50 计划的 `a_ref`。关节与夹爪分别经过幅度、一阶/二阶变化、方向锥、边界跳变和指数滤波约束。
5. Critic 使用双 Q、10-step TD 目标和 `t+10` terminal 合约；每个 episode 由操作者给 reward，并明确选择是否准入训练。候选通过离线验证和 smoke test 后才替换 incumbent，可随时回滚 selector。

更完整的形状、超参数、指标和缓存语义见 `docs/RL_TOKEN_AND_PHASE_CLASSIFIER.md` 与 `docs/RLT_DESIGN.md`。

## 安装布局

为了让现场脚本与成功快照完全一致，推荐保留原目录名：

```bash
git clone <THIS_REPOSITORY_URL> ~/gripper_close_v3_staging_20260727
ln -sfn ~/gripper_close_v3_staging_20260727 ~/piper_jax_inference_v1
cd ~/gripper_close_v3_staging_20260727
uv sync
```

也可以分开维护推理目录和 staging 目录，但修改代码后必须同步两边的 `piper_runtime`。单仓库 + 符号链接可避免版本漂移。当前脚本仍保留现场路径默认值；可用 `RLT_V3_STAGING`、`RLT_V3_WORKSPACE_OVERRIDE`、`RLT_V3_RUNTIME_OVERRIDE`、`OPENPI_WORKSPACE` 等变量覆盖。

另外安装并构建 ROS Noetic、Pika ROS 和 Piper SDK。Pika ROS 安装后应存在：

```text
~/pika_ros/install/setup.bash
~/vendor/piper_sdk_official/piper_sdk
~/venvs/pika/bin/python
```

项目/JAX 与 ROS/硬件 Python 必须分离：JAX 使用仓库 `.venv` 的 Python 3.11，ROS/Piper 使用 `~/venvs/pika` 的 Python 3.8。不要把 ROS Noetic 的 Python 包直接装进 Python 3.11 环境。

## 权重与环境变量

先复制模板并核对：

```bash
cp configs/robot.env.example ~/.config/piper-rlt.env
chmod 600 ~/.config/piper-rlt.env
set -a
source ~/.config/piper-rlt.env
set +a
```

至少准备三类权重：

```text
~/openpi_checkpoints/.../20000/params/                # π0.5 SFT base
~/openpi_rlt/rl_tokens/.../step_010000_encoder_only/ # RL Token encoder
~/rlt_phase_classifiers/.../phase_classifier.pt       # ResNet-18 phase gate
```

在线 RLT Actor checkpoint 位于 session state 的 `learner/step_xxxxxxxx/`；`selected_actor_checkpoint.txt` 决定当前 incumbent。不要把 selector 指向未经验证或不属于同一 lineage 的 checkpoint。

## 上电前只读检查

```bash
ip -details link show can0
rs-enumerate-devices
lsusb
readlink -f /dev/serial/by-path/*
```

期望 `can0` 为 UP、bitrate 1,000,000，且两台 RealSense 的序列号/角色与上表一致。更换相机后必须重新做 exposure、白平衡、视角和 phase-gate 分布检查；不能只改序列号。

启用 CAN 的现场命令：

```bash
cd ~/pika_ros/src/PikaAnyArm/piper/piper_ros
bash can_activate.sh can0 1000000
```

## 数采

原生流程是 CAN → `roscore` → Pika/Sense → Piper teleop → LeRobot record。先按 `data_collection/README.md` 启动 ROS/Pika，再运行：

```bash
cd ~/gripper_close_v3_staging_20260727
bash data_collection/record_lerobot_piper.sh
```

默认是 30 Hz、双相机 640×480@30、`record_only=true`、`action_from_state_delay_s=0.03`、不上传 Hub。task prompt 为 `Put the green block into the box.`。每批数据都应人工抽查相机角色、曝光、action/state 对齐、终止帧和成功标签。

## π0.5 JAX SFT

现场 config 名：`pi05_piper_greenblock_5090_jax_delta_v1`。

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_piper_greenblock_5090_jax_delta_v1
uv run scripts/train.py pi05_piper_greenblock_5090_jax_delta_v1 \
  --exp-name=piper_greenblock_5090_delta_sft_30k_20260707 \
  --fsdp-devices=2
```

现场训练使用 2×A100、batch 32、30,000 steps、EMA 0.999、每 5,000 steps 保存。真机部署使用 step 20,000。不要把旧的“绝对关节动作模型”与本 config 的 delta 训练变换混用。

## 原始 RL Token

RL Token 训练冻结 π0.5 权重，只重建 prefix embeddings：

```bash
uv run scripts/train_rlt_token_jax.py \
  --config-name pi05_piper_greenblock_5090_jax_delta_v1 \
  --checkpoint-dir /path/to/sft/20000 \
  --output-dir /path/to/rl_tokens/run_v1
```

训练完成后导出线上所需的 encoder-only checkpoint：

```bash
uv run scripts/export_rlt_token_encoder_only.py \
  --source /path/to/rl_tokens/run_v1/step_010000 \
  --output /path/to/rl_tokens/run_v1/step_010000_encoder_only
```

原始结构、最终训练指标、验证和缓存注意事项完整记录在 `docs/RL_TOKEN_AND_PHASE_CLASSIFIER.md`。`scripts/train_rlt_token_pytorch.py` 是早期对照实现，不是当前部署权重的来源。

## ResNet 阶段分类器

使用 `configs/phase_annotations_template_30eps.csv` 记录每个 episode 的 `[phase_start_t, phase_end_t]`，再依次执行：

```bash
uv run scripts/build_piper_phase_dataset_from_intervals.py --help
uv run scripts/train_piper_phase_classifier.py --help
uv run scripts/validate_piper_phase_classifier.py --help
uv run scripts/benchmark_phase_classifier_runtime.py --help
```

模板保留了当时 30 条 warm-up 数据的真实 interval，但数据根路径是 5090 快照路径；迁移时必须改成自己的 parquet 路径。禁止提交导出的训练图像和 QA grid，除非已确认无敏感现场内容。

## 纯推理与固定评估

纯推理不需要 Sense：

```bash
bash ~/piper_jax_inference_v1/start_jax_inference_mode.sh
EPISODES=20 bash ~/piper_jax_inference_v1/run_h50_inference.sh
```

固定 Actor checkpoint 评估同样不使用 Sense：

```bash
bash ~/piper_jax_inference_v1/run_fixed_checkpoint_eval_gripper_close_v3.sh \
  --step 821 --episodes 20
```

`run_h50_inference.sh` 每个 episode 先做 4 秒柔和复位；执行 H50，并在动作边界提前 4 步预取。操作者用 `1/0` 记录结果。开始前脚本会拒绝与 Pika 数据采集/相机节点或另一 rollout 并发运行。

## 在线 RLT

当前 fresh-zero lineage 的一键入口：

```bash
bash ~/piper_jax_inference_v1/start_jax_rlt_mode.sh
```

如果没有复制 5090 的 session state，应先创建新的、完全独立的 fresh-zero lineage，而不是伪造 selector：

```bash
bash ~/gripper_close_v3_staging_20260727/run_greenblock_gripper_close_v3_fresh_warmup.sh \
  --name my_gripper_rlt_v1 \
  --warmup-episodes 40 \
  --state-dir .online_rlt_my_gripper_rlt_v1
```

它默认使用：

```text
lineage  = greenblock_rlt_fresh_v3_20260729
state    = .online_rlt_persistent_gripper_v3_fresh
warm-up  = 40 admitted episodes
H50/C10  = base horizon 50, residual chunk 10, stride 10, n-step 10
UTD      = 1
batch    = 256
```

episode 完成后的交互语义：

- `t` / `y`：准入当前 episode，并运行在线 A–C update。
- `r` / `n`：不准入，保留相同 Actor 重试。
- `e` / `q`：不准入并停止。

reward 与“是否准入”是两件事。真实失败模式可以给 reward 0 并准入，给 Critic 提供边界信息；相机异常、夹爪/场景未复位、误操作等 out-of-distribution 事故不要准入。任务最终完成但经历二次抓取，应按预先固定的评估协议标注，不要临时改口径。

在线缓存会先过滤不能构成严格 C10/`t+10` 的行，再通过一个 WebSocket 批请求逐样本保持 batch-size-one 的 frozen encoder 数值路径；manifest 与内容哈希匹配时只增量计算新 episode。不能为了速度取消同计划 C10、精确执行时 `a_ref` 或 terminal 合约。

截至快照：fresh-zero lineage 已训练 119 个 admitted episodes、821 个训练 transitions，当前 selector 为 step 821，最后一次更新新增 5 steps。该状态只作为审计记录，不随 Git 分发。详见 `artifacts/metadata/rlt_step_00000821.json`。

## 测试

```bash
uv run pytest tests/test_rlt_phase_gate.py src/openpi/rlt/real/phase_classifier_test.py
uv run pytest tests/test_pure_inference_wrappers.py tests/test_policy_hardware_rollout.py
uv run pytest scripts/piper_rlt
```

shell 静态检查：

```bash
find . -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

真机测试顺序必须是：只读枚举 → dry-run/协议 smoke → policy server 常驻检查 → CAN 只读状态 → 低幅度单步动作 → 单 episode → 多 episode。不要从“测试通过”直接推导为“可以无人值守”。

日志轮转、统一健康检查和有人见证的故障退出验收见 `docs/OPERATIONS.md`。这些工具只读或旁路运行，不改变训练与动作链。

## 已知限制

- 2026-08 更换腕部 D405 后出现的曝光/域偏移已在现场适配中解决；以后再次更换相机时仍需重新执行曝光、视角和 phase-gate 分布验收。
- `deploy/systemd/5090-snapshot/` 是成功主机的精确路径快照；可移植的健康检查与日志轮转 unit 位于 `deploy/systemd/portable/`。
- 仓库不包含 RLT 论文、第三方 SDK 源码、模型权重或真实数据。

## 许可证与来源

本仓库基于 Physical Intelligence OpenPI 代码并保留 Apache-2.0 与 Gemma 相关许可证；Piper SDK、Pika ROS 和 LeRobot 按各自上游许可单独获取。自定义 Piper/RLT 文件的来源和快照边界见 `docs/SOURCE_PROVENANCE.md`。
