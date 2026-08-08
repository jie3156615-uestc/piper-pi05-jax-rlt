# 5090 上线运维与故障验收

这些工具不修改 Actor、Critic、动作、奖励或 replay。健康检查和默认验收只读，不向机械臂发命令。

## 安装

```bash
cd ~/piper_jax_inference_v1
bash deploy/ops/install_ops.sh
```

安装器会先备份已有同名配置，再安装用户级 systemd unit。日志轮转默认启用；健康检查 timer 默认不启用，避免硬件关机时持续产生失败记录。需要无人值守监控时使用：

```bash
bash deploy/ops/install_ops.sh --enable-health-timer
```

日志达到 100 MiB 或每日检查时轮转，保留 14 份并压缩。`episode.jsonl`、`report.json`、replay、checkpoint 和 `online_state.json` 不轮转。

## 健康检查

RLT 启动完毕后运行：

```bash
python3 scripts/ops/piper_rlt_healthcheck.py --profile rlt
```

退出码：0=全部正常，1=仅警告，2=上线条件失败。机器可离线执行：

```bash
python3 scripts/ops/piper_rlt_healthcheck.py --profile offline --json
```

最近一次定时结果位于 `~/.local/state/piper-rlt/health-latest.json`。

更换 RealSense 后，可在运行前固定预期序列号：

```bash
export RLT_EXPECTED_REALSENSE_SERIALS="<global_serial>,<wrist_serial>"
```

## 无动作软件验收

```bash
bash scripts/ops/run_ops_acceptance.sh offline
bash scripts/ops/run_ops_acceptance.sh rlt
```

第二条只在 RLT 服务已启动后执行。它检查 CAN、相机、Sense、端口、systemd、selected Actor、最近奖励、validation、GPU、磁盘和日志轮转配置，但不会改变服务或发动作。

## 有人见证的物理停止验收

每次硬件/控制器大改后执行一次，并保存终端输出与 `health-latest.json`：

1. 清空机械臂工作区，确认实体急停可触达；Actor 保持当前已验收 checkpoint。
2. 运行一个低幅度测试回合；动作阶段按 `q`，机械臂必须停止，当前回合必须标记为排除训练。
3. 回合间执行 `bash ~/piper_jax_inference_v1/stop_all_rlt_ros.sh`，确认不再有控制命令发布。
4. 人工拔掉一台相机后重新启动，不得进入动作阶段；恢复相机后重新运行完整健康检查。
5. CAN 或 `/arm_status` 异常时不得忽略错误码继续执行；复位故障后从唯一正式入口重启。
6. 检查最后一个 `episode.jsonl`、`report.json` 和服务日志，确认停止原因可追溯，且 selected Actor 未因中止回合改变。

只有以上六项通过，才能把“软件测试通过”提升为“当前硬件组合可上线”。
