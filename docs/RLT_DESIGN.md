# Online RLT 设计与数据合约

## 任务目标

基础 π0.5 完成长时程抓取/放置。在线 RLT 不全程替代 VLA，只在 ResNet 阶段门控进入精细阶段后，以低幅度残差修正夹取/放置轨迹。目标是提高严格成功率，同时保持原 SFT 行为分布和可回滚性。

## 动作坐标

- π0.5 生成 H50、7 维动作参考；执行坐标是绝对关节角 + 绝对夹爪开度。
- RLT Actor 参数化为 C10 residual；关节为 rank-1 bump，夹爪为 close-only knot。
- `chunk_length=10`、`chunk_stride=10`、`n_step=10`。训练 transition 只从满足同一 base plan、精确执行时 `a_ref`、连续 C10 和有效 `t+10` 的物理行构建。
- 因此“物理动作行数”远大于“TD-C10 transitions”是预期行为，不等于丢失数据。验证 split 也从严格 transitions 中划出，而不是从原始动作行直接划出。

## Actor 安全投影

关节残差限制：

```text
|r| <= 0.005 rad
|d1| <= 0.0015 rad/step
|d2| <= 0.001 rad/step²
direction cone <= 15 deg
chunk boundary jump <= 0.06 rad
projection ramp = 33 steps, minimum scale = 0.2
```

夹爪 close-only 限制：

```text
|r| <= 0.005 m
|d1| <= 0.0005 m/step
|d2| <= 0.0003 m/step²
boundary jump <= 0.0005 m
command range = [0.0, 0.08] m
release reference = 0.05 m, release delta guard = 0.002 m
```

执行滤波为 `alpha = 1-exp(-dt/tau)`，`tau=0.05 s`、`dt=1/30 s`，现场 alpha 为 0.486582880967408。

## Actor–Critic

- Actor/Critic learning rate：3e-4；hidden 256；projection 128。
- 双 Q Critic，`gamma=0.99`、target `tau=0.005`、policy delay 2。
- target policy noise std/clip：0.1/0.2；reference dropout 0.5。
- batch 256，UTD 1，gradient clip 10，weight decay 1e-4。
- base BC 权重 20；human joint BC 为 0；human gripper BC 为 1。
- human gripper 候选保留 reward 1/0，使用 `critic_min_advantage_v1`、margin 0 的 Q-filter。

Critic 学习的是 TD（temporal difference，时序差分）目标：当前 Q 应接近 n-step reward 加折扣后的目标 Q。TD error 是预测 Q 与该目标之间的差。平均 TD error 下降只表示拟合更一致，必须与 reward1/reward0 Q gap、双 Q 分歧、Actor advantage、严格真机成功率和动作平滑度一起看。

## Warm-up 与人工标签

fresh-zero lineage 使用 40 个 admitted episodes 后开始正式 A–C 更新。warm-up 的重点不是只收成功，而是覆盖任务边界且标签可靠：

- 严格成功给 reward 1。
- 真实任务失败给 reward 0；只要硬件、相机、复位和操作流程正常，可以准入以帮助 Critic 学边界。
- 相机失真、USB 丢帧、场景未复位、误触、急停、机械故障等外部事故不代表策略质量，不应准入。
- “第一次未夹到、第二次夹到并放正”必须按事先固定口径；如果目标是一次抓取严格成功，应记 0。不要因为结果看起来不错临时把口径改成 1。
- 放置稍有回拉/歪斜是否失败，应由固定容差和终态测量决定；不要担心 reward 0 会“破坏分布”而篡改标签。

## 缓存

RL-token cache 保存每个有效物理行对应的 frozen `z_rl`，避免每次训练重复跑昂贵的 π0.5 prefix encoder。当前实现：

1. 先过滤明显不可能形成严格 C10/`t+10` 的行。
2. 通过单个 WebSocket 批请求降低连接和序列化开销。
3. 每个样本仍走 batch-size-one encoder，避免更改 GPU reduction order 导致 frozen feature 发生可测偏移。
4. 以 episode、输入身份和内容哈希维护 manifest，只计算新增或失效行。
5. replay 仍以精确执行时 `a_ref` 和 fresh per-row `z_rl` 对齐。

缓存是推理预处理，不会向机器人发动作。不能把 `4,270` 个物理动作行直接解释为 `427` 个训练 transition；严格边界、同计划、步长、终止、数据完整性和验证划分都会进一步筛选。

## 候选晋升与回滚

在线更新写入新 checkpoint 后，先运行离线 validation，再重启 shadow policy service 并 smoke test。只有全部通过才原子更新 `selected_actor_checkpoint.txt`。incumbent 不被覆盖；回滚只需让 selector 重新指向同 lineage 的旧 checkpoint，然后重启 shadow service。

不要跨 action schema、base checkpoint、RL-token、phase classifier 或 normalization lineage 直接切换 Actor。所有身份由 metadata 指纹校验。
