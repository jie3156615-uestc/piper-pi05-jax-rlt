# Piper RLT Replay Enrichment

本工具只处理离线数据，不连接 ROS，也不发布机械臂命令。

## 生产用法

先停止原 8000 policy service，再单独启动 8001 shadow service，避免 5090 同时加载两份
大模型导致 OOM。cache 生成器不会导入 ROS/CAN，也不会发布动作：

```bash
systemctl --user stop openpi-piper-policy.service
PIPER_RLT_ACTOR_MODE=none \
PIPER_RL_TOKEN_CHECKPOINT=~/openpi_rlt/rl_tokens/pi05_piper_greenblock_5090_full20k_rlt_token_v1_20260709/step_010000_encoder_only \
bash ~/piper_jax_inference_v1/run_rlt_shadow_policy_server.sh
```

另开终端生成可断点续跑的 cache：

```bash
cd ~/openpi_jax_piper_lora_v1_20260707
PYTHONPATH=src:~/piper_jax_inference_v1:packages/openpi-client/src \
.venv/bin/python scripts/piper_rlt/tools/generate_external_rlt_enrichment_cache.py \
  --episode-dir ~/rlt_takeover_sessions \
  --dataset-root ~/rlt_takeover_sessions \
  --output ~/openpi_rlt/replays/greenblock_box_v1/enrichment_cache.jsonl \
  --phase-checkpoint ~/rlt_phase_classifiers/greenblock_box_resnet18_v4_manual_intervals/phase_classifier.pt \
  --policy-port 8001 \
  --base-fingerprint FULL20K_ID_OR_SHA256 \
  --token-fingerprint 2f2e1e6bbcae8f08217ec7ba0b88088bfa44627be495035319deb84e06052b49 \
  --skip-invalid-episodes
```

默认要求每个 episode 同时具有 `report.json`、`outcome=episode_done`、
`terminal_reward=0/1`，因此会排除无 report、reward=null 和 debug episode。
所有 raw row 都计算 Phase；只有 single-latch 已激活且动作行有效时才请求
full-20k/Token。中断时保留 `.partial` journal，重跑同一命令即可续跑；最终
JSONL 和 manifest 都通过原子 rename 提交，并校验输入图像/state 指纹。

生成后直接使用 manifest 中精确选中的 episode（当前预期为 16 个），不要再用
`--episode-dir` 把 debug/nonterminal episode 混回来：

```bash
cd ~/openpi_jax_piper_lora_v1_20260707
PYTHONPATH=src .venv/bin/python scripts/piper_rlt/tools/prepare_external_rlt_replay.py \
  --episode-manifest ~/openpi_rlt/replays/greenblock_box_v1/enrichment_cache.jsonl.manifest.json \
  --dataset-root ~/rlt_takeover_sessions \
  --enrichment-cache /path/to/full20k_token_phase_cache.jsonl \
  --output ~/openpi_rlt/replays/greenblock_box_v1 \
  --base-fingerprint FULL20K_SHA256 \
  --token-fingerprint RL_TOKEN_SHA256 \
  --phase-fingerprint PHASE_RESNET_SHA256
```

固定的数据契约是 `C=10, stride=2, n=10`。生产数据不要使用
`--allow-logged-reference` 或 `--allow-logged-phase`；它们仅供 schema/debug
测试。

## Cache 格式

每行是一个 JSON 对象，key 为全局唯一的 `episode_id + t`。当
`--dataset-root=~/rlt_takeover_sessions` 时，episode ID 例如：

```text
rlt_session_20260709_112516/episode_000000
```

Phase 对所有 raw row 计算，因此每行至少需要：

```json
{"episode_id":"rlt_session_x/episode_000000","t":0,"phase_probability":0.12}
```

只有 single-latch gate 已激活且通过 source/replay/wait 过滤的动作行，才需要
full-20k reference 和 RL Token：

```json
{
  "episode_id":"rlt_session_x/episode_000000",
  "t":120,
  "phase_probability":0.91,
  "a_ref":[[0.01,0.02,0.03,0.04,0.05,0.06,0.04]],
  "z_rl":[0.1,0.2],
  "action_space":"joint_delta_gripper_absolute"
}
```

`a_ref` 可以有任意正 horizon，输出会裁剪/末值补齐到 10。支持：

- `joint_delta_gripper_absolute`：前六维相对该 row state，第七维绝对夹爪；
- `joint_absolute_gripper_absolute`：七维均为绝对发布目标。

也可使用 `--reference-provider package.module:function` 和
`--phase-provider package.module:function`。回调签名均为
`(record: RealStepRecord, episode_root: Path)`；reference 返回
`ReferenceTokenValue` 或与上述 cache 相同的 mapping，phase 返回 float 或
`{"phase_probability": float}`。

## 输出语义

- 严格排除 `stop/safety_block/wait/hold`、`replay_include=false` 和奖励等待行；
- gate 未进入前不调用 full-20k/Token provider，也不进入 replay；
- gate 一旦进入，在 episode terminal 前不会退出；
- terminal `reward/done` 迁移到最后一个有效执行动作；
- 无效行形成硬 segment 边界，chunk 不能跨越；
- `a_ref/a_exec/a_human/a_actor` 的前六维是相对 chunk 起始 state 的 delta，
  第七维是绝对夹爪；
- `*_absolute` 和 `a_ref_original_absolute` 用于审计；
- `source_chunk/human_mask/actor_mask/step_mask` 明确区分动作来源、缺失值和 padding。
