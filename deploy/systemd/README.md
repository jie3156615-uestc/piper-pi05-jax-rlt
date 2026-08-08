# systemd 用户服务

`5090-snapshot/` 是成功运行主机 `cwzk-MS-7D99` 的精确用户服务快照，包含 `/home/cwzk/...` 路径。保留这些文件是为了审计实际部署，不应在另一台机器上直接复制启用。

迁移步骤：

1. 将 repo、Pika ROS、两个 Python 环境和模型放到目标路径。
2. 复制需要的 unit 到 `~/.config/systemd/user/`。
3. 修改 `WorkingDirectory`、`PYTHONPATH`、`ExecStart`、checkpoint、selector、Sense by-path。
4. `systemd-analyze --user verify ~/.config/systemd/user/*.service`。
5. `systemctl --user daemon-reload`，只启动 `rlt-roscore.service` 做检查。
6. 确认纯推理与 RLT 不会同时拥有同名 Piper ROS controller，再逐个启动其余服务。

`run_rlt_lineage_gripper_close_v3_online.sh` 会根据当前 lineage 动态生成 gripper-v3 shadow unit 的关键环境，因此在线训练应优先通过受保护的 launcher 启动。
