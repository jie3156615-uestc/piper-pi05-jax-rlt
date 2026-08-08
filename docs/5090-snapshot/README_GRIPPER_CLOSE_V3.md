# 5090 Piper RLT 最小启动与续训说明

更新时间：2026-07-28（Asia/Shanghai）

## 1. 当前默认实验

```text
lineage: greenblock_rlt_gripper_close_v3_restart_from_bootstrap_20260728
state:   .online_rlt_persistent_gripper_v3
Actor:   step_00000288（已验收）
first:   episode_000408
```

这是独立的 clean-restart 分支：

- 复用已经审计的30轮 warmup，不需要重新采集。
- 复用已验收 `step_00000288`。
- 不带入旧分支5次被验收门拒绝的候选更新。
- 不使用 `learner/latest.txt`。旧分支的 `step_00000345` 是被拒候选，不是部署 Actor。

## 2. 最简启动

旧会话必须先在原终端正常退出。确认没有 rollout/learner：

```bash
pgrep -af '[r]lt_online_session|[r]lt_takeover_rollout|[t]rain_real_rlt_jax|[r]un_online_rlt_update'
```

无输出后，先做只读预检：

```bash
bash ~/gripper_close_v3_staging_20260727/run_greenblock_gripper_close_v3_current.sh \
  --dry-run
```

正式启动：

```bash
bash ~/gripper_close_v3_staging_20260727/run_greenblock_gripper_close_v3_current.sh \
  --actor-live-max-chunks 0
```

`0`表示关键阶段内不限制 Actor C10 数量；关节、速度、加速度、边界连续性、夹爪范围和人工接管保护仍然有效。

正式入口还带有跨谱系全局进程门：只要另一条 rollout 或 learner 仍在运行，新启动就会在任何ROS/CAN/机械臂修改前拒绝。不同谱系不能并行控制机械臂。

## 3. 如何指定 episode 继续同一谱系

推荐让脚本自动读取磁盘最新完整轮次：

```bash
bash ~/gripper_close_v3_staging_20260727/run_greenblock_gripper_close_v3_current.sh \
  --latest-episode AUTO \
  --actor-live-max-chunks 0
```

首次运行新分支时没有 episode 目录，也可显式写：

```bash
bash ~/gripper_close_v3_staging_20260727/run_greenblock_gripper_close_v3_current.sh \
  --lineage greenblock_rlt_gripper_close_v3_restart_from_bootstrap_20260728 \
  --state-dir .online_rlt_persistent_gripper_v3 \
  --latest-episode NONE \
  --actor-live-max-chunks 0
```

以后若磁盘最新完整轮次例如为 `episode_000421`：

```bash
bash ~/gripper_close_v3_staging_20260727/run_greenblock_gripper_close_v3_current.sh \
  --lineage greenblock_rlt_gripper_close_v3_restart_from_bootstrap_20260728 \
  --state-dir .online_rlt_persistent_gripper_v3 \
  --latest-episode episode_000421 \
  --actor-live-max-chunks 0
```

这里的 `--latest-episode` 只是并发/误启动守卫：

- 必须等于该谱系磁盘上的最新完整目录。
- 下一轮始终是最新目录加一。
- 它不选择 Actor、不选择 checkpoint，也不能把现有谱系回退到历史轮次。
- 传旧编号会被拒绝，不会覆盖数据。

三者不要混淆：

```text
最新 episode 目录  = rollout编号与并发守卫
人工准入 episode   = replay是否纳入候选训练
promoted checkpoint = 真正用于下一轮推理的Actor
```

## 4. 继续旧分支

旧分支保留不动：

```text
lineage: greenblock_rlt_gripper_close_v3_from_v2_ep407_20260727
state:   .online_rlt_persistent_gripper_v3
```

2026-07-28审计边界为 `episode_000442`，下一轮为 `000443`，部署仍是 `step_00000288`。若该旧会话之后又产生新目录，请使用 `AUTO`，不要照抄442：

```bash
bash ~/gripper_close_v3_staging_20260727/run_greenblock_gripper_close_v3_current.sh \
  --lineage greenblock_rlt_gripper_close_v3_from_v2_ep407_20260727 \
  --state-dir .online_rlt_persistent_gripper_v3 \
  --latest-episode AUTO \
  --actor-live-max-chunks 0
```

旧分支5个入训候选的映射为：

```text
episode_000413 -> step_00000299 -> rejected
episode_000416 -> step_00000309 -> rejected
episode_000417 -> step_00000320 -> rejected
episode_000424 -> step_00000335 -> rejected
episode_000442 -> step_00000345 -> rejected
```

因此旧分支没有在408以后产生新的 promotion。

## 5. 另起分支重新开始

当前 clean-restart 分支已经创建，不要再次创建同名目录。

以后需要再建一个独立分支时，先只读预检：

```bash
bash ~/gripper_close_v3_staging_20260727/create_greenblock_gripper_close_v3_restart_branch.sh \
  --target-lineage 自定义的新谱系名
```

确认预检通过后才真正创建：

```bash
bash ~/gripper_close_v3_staging_20260727/create_greenblock_gripper_close_v3_restart_branch.sh \
  --target-lineage 自定义的新谱系名 \
  --create
```

该工具只做“同一30轮bootstrap + 已验收step288”的干净重开，首轮固定为 `episode_000408`；它不启动ROS/CAN/机械臂，也不重新训练初始模型。

不能把任意历史 rollout 目录直接当作 checkpoint。若未来要从历史 Actor 另起分支，只能选择当时已经正式 promotion 的边界，并复制与校验其 checkpoint、replay、hash和state。当前旧v3在408以后没有 promotion，所以从413、416、417、424或442“选模型”得到的仍只能是step288。

## 6. Rollout后的选择

输入 `1/0` 后：

```text
t / y = 本轮准入训练，运行A-C候选更新与验收
r / n = 本轮不训练，用同一promoted Actor再测
e / q = 本轮不训练并停止
```

只有候选通过 validation/promotion 后，下一轮 Actor 才会改变。训练运行很快不等于已部署；以以下三个位置为准：

```text
state/online_state.json: latest_checkpoint / deployment_checkpoint
state/selected_actor_checkpoint.txt
启动预检打印的 promoted Actor
```

## 7. 只读核验

```bash
bash ~/gripper_close_v3_staging_20260727/run_greenblock_gripper_close_v3_current.sh \
  --lineage greenblock_rlt_gripper_close_v3_restart_from_bootstrap_20260728 \
  --state-dir .online_rlt_persistent_gripper_v3 \
  --latest-episode NONE \
  --dry-run
```

预期显示：

```text
newest completed dir  : NONE
next rollout          : episode_000408
promoted Actor        : .../step_00000288
trained episodes      : 30
accepted update index : 1
```

`--dry-run`结束时必须明确说明没有修改ROS、selector、CAN或机械臂命令。
