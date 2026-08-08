# 原始 RL Token 与 ResNet 阶段分类器

## JAX RL Token

### 作用

原始 RL Token 是对冻结 π0.5 prefix embeddings 的紧凑任务/视觉/状态表示。它供 RLT Actor 和 Critic 使用，但不替换、删除或重训 VLA 自己的视觉 token。在线 policy 仍先走完整 π0.5 观察变换与 prefix encoder。

### 已部署结构

```text
input prefix embeddings : [B, T, 2048]
learned RL token        : 1
encoder layers          : 2
decoder layers          : 2（只用于训练重建）
attention heads         : 8
MLP ratio               : 4.0
dropout                 : 0.0
max sequence length     : 1024
runtime z_rl            : [B, 2048]
```

训练脚本是 `scripts/train_rlt_token_jax.py`，模型定义是 `src/openpi/rlt/jax_token.py`。它冻结 step-20k π0.5，只优化 autoencoder；训练 10,000 steps、batch 4、AdamW、lr 1e-4、seed 42、num_workers 0，每 1,000 steps 保存。训练后可生成 `cache_z_rl`，缓存 batch 为 8。

step 10,000 的最终记录：

| 指标 | 值 |
| --- | ---: |
| reconstruction loss | 0.1067691892 |
| gradient norm | 0.3231828511 |
| mean valid prefix tokens | 522 |
| mean `z_rl` norm | 38.4638977051 |

完整 autoencoder 格式为 `jax_flax_rl_token_autoencoder_v1`。线上通过 `scripts/export_rlt_token_encoder_only.py` 只保留 `rl_token`、`encoder_pos`、encoder blocks 和 `z_norm`，格式为 `jax_flax_rl_token_encoder_v1`。现场 encoder-only SHA-256 身份指纹：

```text
2f2e1e6bbcae8f08217ec7ba0b88088bfa44627be495035319deb84e06052b49
```

注意 `scripts/train_rlt_token_pytorch.py` 是早期 PyTorch 对照，当前 RLT 使用的是上述 JAX/Flax checkpoint，不能互换权重。

### 验证与数值一致性

- `scripts/validate_rlt_token_jax.py`：基本结构、加载与重建检查。
- `scripts/validate_rlt_token_jax_dataset.py`：数据集分布检查。
- `scripts/validate_rlt_token_jax_heldout.py`：held-out 检查。
- `scripts/benchmark_rlt_service_protocol.py`：单条/批协议延迟与值一致性。
- `piper_runtime/rlt_token_runtime.py`：线上 encoder-only 加载与 per-item batch-size-one 语义。

批 WebSocket 是“网络批处理”，不是把全部图像堆成一个更大的 JAX batch。后者会改变 reduction order，并使现有 cache 与新计算的 frozen feature 有可测差异。保持 per-item 路径是为了精度与 lineage 一致性。

## ResNet-18 阶段分类器

### 作用与输入

小网络只判断何时进入精细插入/夹取/放置阶段。输入不是 π0.5 token，而是两路 RGB：`camera1` 和 `camera2` 各缩放到 320×240，横向拼接为 640×240，再 resize 到 224×224，使用 ImageNet mean/std 归一化。

线上门控：threshold 0.5，连续 3 帧达到阈值后进入 `ACTIVE`，单次锁存直到 terminal，不依据概率反复退出/进入。

### 标注和训练

`configs/phase_annotations_template_30eps.csv` 记录 30 个 warm-up episodes 的手工 `[phase_start_t, phase_end_t]`。训练数据为 1,458 正帧、2,952 负帧，`pos_weight=2.0246913580`。ResNet-18 使用 ImageNet pretrained 初始化，CPU、8 epochs、batch 128、lr 3e-4。

每轮训练 loss：

```text
0.113196, 0.033195, 0.026052, 0.024750,
0.017075, 0.019419, 0.014100, 0.006045
```

7 个 held-out episodes、1,293 帧的旧相机分布验证结果：

| 指标 | 值 |
| --- | ---: |
| accuracy | 0.972931 |
| precision | 0.961538 |
| recall | 0.959368 |
| F1 | 0.960452 |
| FPR / FNR | 0.020000 / 0.040632 |
| TP / FP / TN / FN | 425 / 17 / 833 / 18 |

checkpoint 身份指纹：

```text
8c5b443edd3f399529680ef5e4c5dffcdee4af2ae2f224f6da4152237c9a50dc
```

### 换相机后的强制检查

更换为新 D405 后，序列号适配并不代表数据分布适配。已观察到腕部图像均值从约 120 升到 204–214，饱和像素从接近 0 升到 32%–56%，phase probability 可降到约 0.008。此时 Actor 不进入并不是 Actor checkpoint 失效，而是 phase gate 输入域偏移。

恢复训练前必须：锁定曝光/白平衡，复现旧相机直方图和视角；运行 `smoke_phase_classifier_runtime.py`、`check_phase_gate_timeline.py` 和 held-out + 新相机校验；对 phase 触发时刻做人工视频抽查。异常 episode 不应准入在线 RLT。
