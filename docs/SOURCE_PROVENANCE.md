# 代码来源与快照边界

- `src/openpi/`、`packages/openpi-client/`、`pyproject.toml`、`uv.lock` 以 Physical Intelligence OpenPI 为基础，叠加 Piper 数据变换、JAX RL Token 和 real-world RLT 代码。
- `piper_runtime/` 是 5090 真机运行层：RealSense、ROS/CAN 状态、policy client/server、Pika/Sense takeover、phase gate、RL-token runtime、Actor protocol、残差 governor、episode logger。
- `scripts/piper_rlt/tools/` 是在线 replay、严格合约审计、增量 enrichment、训练、验证、lineage 初始化/迁移和 checkpoint 晋升工具。
- 根目录 `run_*`/`start_*` 是 2026-07-31 至 2026-08-08 期间实际使用的启动器。`docs/5090-snapshot/` 和 `deploy/systemd/5090-snapshot/` 保留成功主机的路径化快照，迁移时不能直接假设路径仍正确。
- `pika_overrides/` 与 `ros/` 是本项目对 Pika/Piper ROS 的小范围覆盖和 launch 文件；完整 Pika ROS 与 Piper SDK 不在本仓库中。

上游版本基线：

```text
piper_sdk  c05c5454b1cf61c05ad26385e0c0a3aa6d3c7bad
pika_ros   f40dc2868d87a6285d60524c37733101aa5785ee
lerobot checkout on 5090  a5b29d430105f5235eb05bbf2db5a0d747a869d6
lerobot dependency pin   0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
```

排除项：模型/优化器权重、真实数据、论文 PDF、构建产物、ROS install tree、日志、回放、token cache、相机 QA 图、临时备份和旧实验分支脚本。
