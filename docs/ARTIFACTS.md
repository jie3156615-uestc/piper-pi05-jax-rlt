# 外部模型与数据放置约定

Git 仓库不分发权重和真实数据。下面是 5090 快照中的身份与推荐目录。

| Artifact | 参考路径 | 身份/说明 |
| --- | --- | --- |
| π0.5 base | `~/openpi_checkpoints/pi05_piper_greenblock_5090_jax_delta_v1/piper_greenblock_5090_delta_sft_30k_20260707/20000` | `full20k_step20000_metadata_sha256_14d9cac129ec7ce91f2e5aab3f5bfac06172c8fb70709f01850fb8e8215870e5` |
| RL Token full | `~/openpi_rlt/rl_tokens/pi05_piper_greenblock_5090_full20k_rlt_token_v1_20260709/step_010000` | 训练/研究用 autoencoder，约 973 MiB |
| RL Token encoder | 同目录 `step_010000_encoder_only` | 线上用，约 411 MiB；fingerprint `2f2e1e6b...` |
| Phase classifier | `~/rlt_phase_classifiers/greenblock_box_resnet18_v4_manual_intervals/phase_classifier.pt` | ResNet-18，约 44.8 MiB；fingerprint `8c5b443e...` |
| Online state | `~/rlt_online_sessions/greenblock_rlt_fresh_v3_20260729/.online_rlt_persistent_gripper_v3_fresh` | fresh-zero lineage；selector、replay、cache、learner checkpoint |

复制后至少检查：

```bash
test -d "$PIPER_POLICY_CHECKPOINT/params"
test -f "$PIPER_RL_TOKEN_CHECKPOINT/params.msgpack"
test -f "$PIPER_RL_TOKEN_CHECKPOINT/metadata.json"
test -f "$PIPER_PHASE_CHECKPOINT"
```

如需在 GitHub 分发权重，使用单独的私有 Release、Git LFS 或模型仓库，并先核对许可证。不要把真实 episode、现场图像、session cache 或带绝对路径的完整日志一起上传。

`artifacts/metadata/` 中的小 JSON 是人工整理的审计摘要，不含权重，也不能单独恢复模型。
