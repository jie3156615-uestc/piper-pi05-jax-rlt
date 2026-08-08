# Piper + 蟺0.5 JAX + Online RLT

杩欐槸鍗曡噦 AgileX Piper 鍦ㄧ湡瀹炵‖浠朵笂杩愯 蟺0.5銆丣AX SFT 鍜屽湪绾?RLT 鐨勫彲瀹¤浠ｇ爜蹇収銆備粨搴撴暣鐞嗚嚜 5090 涓绘満涓婂凡鎴愬姛杩愯鐨勬暟閲囥€丼FT銆佺函鎺ㄧ悊銆丷L Token銆丷esNet 闃舵闂ㄦ帶鍜?Actor鈥揅ritic 鍦ㄧ嚎鏇存柊閾捐矾锛涘揩鐓ф棩鏈熶负 **2026-08-08**銆?
> **瀹夊叏鎻愮ず**锛氫粨搴撲細鍚戠湡瀹炴満姊拌噦鍙戞寚浠ゃ€傜涓€娆￠儴缃插彧鍋氬彧璇绘鏌ュ拰 dry-run锛涚‘璁ゆ€ュ仠銆丆AN銆佺浉鏈烘槧灏勩€佸叧鑺傛柟鍚戙€佸伐浣滅┖闂村拰鐜板満浜哄憳鍧囧畨鍏ㄥ悗锛屾墠鑳戒娇鐢ㄥ甫 `I_UNDERSTAND_POLICY_MOVES_ARM` 鐨勬墽琛屽叆鍙ｃ€傞獙璇佹寚鏍囦笉鏄墿鐞嗗畨鍏ㄨ瘉鏄庛€?
## 浠撳簱鍐呭

| 妯″潡 | 涓昏鍏ュ彛 | 璇存槑 |
| --- | --- | --- |
| LeRobot 鏁伴噰 | `data_collection/record_lerobot_piper.sh` | Piper 鍘熺敓鐘舵€?鍔ㄤ綔銆丳ika/Sense 浜哄伐绀烘暀銆佸弻 RealSense銆?0 Hz |
| 蟺0.5 SFT | `scripts/train.py`銆乣src/openpi/training/config.py` | 50 姝ュ姩浣滃潡锛涘墠 6 缁村叧鑺備负鐩稿棣栧抚鐨?delta锛屽す鐖繚鎸佺粷瀵归噺 |
| 绾帹鐞?| `start_jax_inference_mode.sh`銆乣run_h50_inference.sh` | policy server 甯搁┗锛汬50 鎵ц锛涙棤闇€ Sense锛? 绉掓煍鍜屽浣?|
| 鍥哄畾杞璇勪及 | `run_fixed_checkpoint_eval_gripper_close_v3.sh` | 鍥哄畾 Actor checkpoint 鐨勫 episode 鐪熸満璇勪及 |
| 鍘熷 RL Token | `scripts/train_rlt_token_jax.py`銆乣src/openpi/rlt/jax_token.py` | 鍐荤粨 蟺0.5 鍓嶇紑 token锛岃缁?JAX/Flax autoencoder锛屽啀瀵煎嚭 encoder-only |
| ResNet 闃舵鍒嗙被鍣?| `scripts/build_piper_phase_dataset_from_intervals.py`銆乣scripts/train_piper_phase_classifier.py` | 鍒ゆ柇浣曟椂杩涘叆绮剧粏澶瑰彇/鏀剧疆闃舵锛屼笉鏇挎崲 VLA 瑙嗚 token |
| 鍦ㄧ嚎 RLT | `start_jax_rlt_mode.sh`銆乣run_rlt_lineage_gripper_close_v3_online.sh` | H50 base + C10 娈嬪樊 Actor銆佸弻 Q Critic銆佷汉宸ュ鍔?鍑嗗叆銆佷弗鏍?replay 鍚堢害 |
| 缂撳瓨鍜屽璁?| `scripts/piper_rlt/tools/` | 澧為噺 RL-token cache銆佺簿纭墽琛屾椂 `a_ref`銆佸悓璁″垝 C10銆乣t+10`銆佸€欓€夐獙璇?鍥炴粴 |

妯″瀷鏉冮噸銆佺湡瀹?episode銆佸浘鍍忋€佽棰戙€乺eplay銆丷L-token cache銆佽缁冩棩蹇楀拰鍦ㄧ嚎 session **涓嶅湪 Git 涓?*銆傚畠浠綋绉ぇ涓斿彲鑳藉寘鍚幇鍦轰俊鎭紱浠撳簱浠呮彁浜や唬鐮併€侀厤缃ā鏉裤€佹寚鏍囧拰 SHA-256 鎸囩汗銆傚畬鏁存斁缃害瀹氳 `docs/ARTIFACTS.md`銆?
## 鍙傝€冪‖浠朵笌杞欢

杩欎笉鏄渶浣庨厤缃紝鑰屾槸宸查獙璇佺殑 5090 涓绘満蹇収銆?
| 椤圭洰 | 宸查獙璇侀厤缃?|
| --- | --- |
| 涓绘満 | `cwzk-MS-7D99`锛孶buntu 20.04.6 LTS锛孡inux 5.15.0-139锛寈86_64 |
| CPU / 鍐呭瓨 | Intel Core i7-12700F锛?1 GiB RAM锛? GiB swap |
| GPU | NVIDIA GeForce RTX 5090 D v2锛?4,455 MiB锛宒river 580.126.09锛孋UDA compute capability 12.0 |
| 鏈烘鑷?| 鍗曡噦 AgileX Piper锛孋AN `can0`锛?,000,000 bit/s |
| 绀烘暀鍣?| Pika/Sense锛涗粎鏁伴噰涓?RLT 浜哄伐浠嬪叆浣跨敤锛岀函鎺ㄧ悊涓嶄娇鐢?|
| 鍏ㄥ眬鐩告満 | Intel RealSense D435锛屽簭鍒楀彿 `347522072112`锛岃繍琛屾椂閿?`camera1` |
| 鑵曢儴鐩告満 | Intel RealSense D405锛屽簭鍒楀彿 `260622272544`锛岃繍琛屾椂閿?`camera2` |
| ROS | ROS 1 Noetic |
| JAX 鐜 | Python 3.11.15锛汮AX/JAXlib 0.5.3锛孎lax 0.10.2锛孫ptax 0.2.8锛孫rbax 0.11.13 |
| 纭欢/ROS 鐜 | Python 3.8.10锛沺iper-sdk 0.6.1锛宲ython-can 4.5.0锛宲yrealsense2 2.55.1.6486 |

宸查獙璇佺殑涓婃父鐗堟湰锛?
- `agilexrobotics/piper_sdk`锛歚c05c5454b1cf61c05ad26385e0c0a3aa6d3c7bad`
- `agilexrobotics/pika_ros`锛歚f40dc2868d87a6285d60524c37733101aa5785ee`
- 5090 涓婃鏌ュ埌鐨?LeRobot checkout锛歚a5b29d430105f5235eb05bbf2db5a0d747a869d6`
- 鏈粨搴?`uv.lock`/`pyproject.toml` 鐨?LeRobot 渚濊禆鍥哄畾涓?`0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`

鏈€鍚庝袱椤规槸鈥滅幇鍦?checkout鈥濆拰鈥淥penPI 鍙鐜颁緷璧栤€濅袱涓笉鍚屾蹇碉紝涓嶅簲娣风敤銆?
## 绠楁硶閾捐矾

1. 蟺0.5 SFT 鏍规嵁鍙岀浉鏈恒€? 缁寸姸鎬佸拰鏂囨湰浠诲姟鐢熸垚 `H=50` 鐨勭粷瀵规墽琛屽弬鑰冨姩浣溿€傛暟鎹缁冩椂灏嗗墠 6 涓叧鑺傜淮浠庣粷瀵硅杞负鐩稿鍔ㄤ綔鍧楅鐘舵€佺殑 delta锛涚 7 缁村す鐖繚鎸佺粷瀵归噺銆?2. 鍐荤粨鐨?蟺0.5 prefix embeddings 杈撳叆 RL Token autoencoder銆傚湪绾垮彧鍔犺浇 encoder锛屽緱鍒?2,048 缁?`z_rl`锛涘畠鏄?Actor/Critic 鐨勬潯浠剁壒寰侊紝涓嶅彇浠?蟺0.5 鐨勮瑙?token銆?3. ResNet-18 璇诲彇 `camera1|camera2` 妯悜鎷兼帴鍥撅紝杩炵画 3 甯ф鐜囦笉浣庝簬 0.5 鍚庡崟娆￠攣瀛橈紝鐩村埌 episode 缁堟锛汻LT 鍙湪杩欎釜绮剧粏闃舵浠嬪叆銆?4. Actor 杈撳嚭 10 姝?C10 娈嬪樊锛屽彔鍔犲埌鍚屼竴 H50 璁″垝鐨?`a_ref`銆傚叧鑺備笌澶圭埅鍒嗗埆缁忚繃骞呭害銆佷竴闃?浜岄樁鍙樺寲銆佹柟鍚戦敟銆佽竟鐣岃烦鍙樺拰鎸囨暟婊ゆ尝绾︽潫銆?5. Critic 浣跨敤鍙?Q銆?0-step TD 鐩爣鍜?`t+10` terminal 鍚堢害锛涙瘡涓?episode 鐢辨搷浣滆€呯粰 reward锛屽苟鏄庣‘閫夋嫨鏄惁鍑嗗叆璁粌銆傚€欓€夐€氳繃绂荤嚎楠岃瘉鍜?smoke test 鍚庢墠鏇挎崲 incumbent锛屽彲闅忔椂鍥炴粴 selector銆?
鏇村畬鏁寸殑褰㈢姸銆佽秴鍙傛暟銆佹寚鏍囧拰缂撳瓨璇箟瑙?`docs/RL_TOKEN_AND_PHASE_CLASSIFIER.md` 涓?`docs/RLT_DESIGN.md`銆?
## 瀹夎甯冨眬

涓轰簡璁╃幇鍦鸿剼鏈笌鎴愬姛蹇収瀹屽叏涓€鑷达紝鎺ㄨ崘淇濈暀鍘熺洰褰曞悕锛?
```bash
git clone <THIS_REPOSITORY_URL> ~/gripper_close_v3_staging_20260727
ln -sfn ~/gripper_close_v3_staging_20260727 ~/piper_jax_inference_v1
cd ~/gripper_close_v3_staging_20260727
uv sync
```

涔熷彲浠ュ垎寮€缁存姢鎺ㄧ悊鐩綍鍜?staging 鐩綍锛屼絾淇敼浠ｇ爜鍚庡繀椤诲悓姝ヤ袱杈圭殑 `piper_runtime`銆傚崟浠撳簱 + 绗﹀彿閾炬帴鍙伩鍏嶇増鏈紓绉汇€傚綋鍓嶈剼鏈粛淇濈暀鐜板満璺緞榛樿鍊硷紱鍙敤 `RLT_V3_STAGING`銆乣RLT_V3_WORKSPACE_OVERRIDE`銆乣RLT_V3_RUNTIME_OVERRIDE`銆乣OPENPI_WORKSPACE` 绛夊彉閲忚鐩栥€?
鍙﹀瀹夎骞舵瀯寤?ROS Noetic銆丳ika ROS 鍜?Piper SDK銆侾ika ROS 瀹夎鍚庡簲瀛樺湪锛?
```text
~/pika_ros/install/setup.bash
~/vendor/piper_sdk_official/piper_sdk
~/venvs/pika/bin/python
```

椤圭洰/JAX 涓?ROS/纭欢 Python 蹇呴』鍒嗙锛欽AX 浣跨敤浠撳簱 `.venv` 鐨?Python 3.11锛孯OS/Piper 浣跨敤 `~/venvs/pika` 鐨?Python 3.8銆備笉瑕佹妸 ROS Noetic 鐨?Python 鍖呯洿鎺ヨ杩?Python 3.11 鐜銆?
## 鏉冮噸涓庣幆澧冨彉閲?
鍏堝鍒舵ā鏉垮苟鏍稿锛?
```bash
cp configs/robot.env.example ~/.config/piper-rlt.env
chmod 600 ~/.config/piper-rlt.env
set -a
source ~/.config/piper-rlt.env
set +a
```

鑷冲皯鍑嗗涓夌被鏉冮噸锛?
```text
~/openpi_checkpoints/.../20000/params/                # 蟺0.5 SFT base
~/openpi_rlt/rl_tokens/.../step_010000_encoder_only/ # RL Token encoder
~/rlt_phase_classifiers/.../phase_classifier.pt       # ResNet-18 phase gate
```

鍦ㄧ嚎 RLT Actor checkpoint 浣嶄簬 session state 鐨?`learner/step_xxxxxxxx/`锛沗selected_actor_checkpoint.txt` 鍐冲畾褰撳墠 incumbent銆備笉瑕佹妸 selector 鎸囧悜鏈粡楠岃瘉鎴栦笉灞炰簬鍚屼竴 lineage 鐨?checkpoint銆?
## 涓婄數鍓嶅彧璇绘鏌?
```bash
ip -details link show can0
rs-enumerate-devices
lsusb
readlink -f /dev/serial/by-path/*
```

鏈熸湜 `can0` 涓?UP銆乥itrate 1,000,000锛屼笖涓ゅ彴 RealSense 鐨勫簭鍒楀彿/瑙掕壊涓庝笂琛ㄤ竴鑷淬€傛洿鎹㈢浉鏈哄悗蹇呴』閲嶆柊鍋?exposure銆佺櫧骞宠　銆佽瑙掑拰 phase-gate 鍒嗗竷妫€鏌ワ紱涓嶈兘鍙敼搴忓垪鍙枫€?
鍚敤 CAN 鐨勭幇鍦哄懡浠わ細

```bash
cd ~/pika_ros/src/PikaAnyArm/piper/piper_ros
bash can_activate.sh can0 1000000
```

## 鏁伴噰

鍘熺敓娴佺▼鏄?CAN 鈫?`roscore` 鈫?Pika/Sense 鈫?Piper teleop 鈫?LeRobot record銆傚厛鎸?`data_collection/README.md` 鍚姩 ROS/Pika锛屽啀杩愯锛?
```bash
cd ~/gripper_close_v3_staging_20260727
bash data_collection/record_lerobot_piper.sh
```

榛樿鏄?30 Hz銆佸弻鐩告満 640脳480@30銆乣record_only=true`銆乣action_from_state_delay_s=0.03`銆佷笉涓婁紶 Hub銆倀ask prompt 涓?`Put the green block into the box.`銆傛瘡鎵规暟鎹兘搴斾汉宸ユ娊鏌ョ浉鏈鸿鑹层€佹洕鍏夈€乤ction/state 瀵归綈銆佺粓姝㈠抚鍜屾垚鍔熸爣绛俱€?
## 蟺0.5 JAX SFT

鐜板満 config 鍚嶏細`pi05_piper_greenblock_5090_jax_delta_v1`銆?
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_piper_greenblock_5090_jax_delta_v1
uv run scripts/train.py pi05_piper_greenblock_5090_jax_delta_v1 \
  --exp-name=piper_greenblock_5090_delta_sft_30k_20260707 \
  --fsdp-devices=2
```

鐜板満璁粌浣跨敤 2脳A100銆乥atch 32銆?0,000 steps銆丒MA 0.999銆佹瘡 5,000 steps 淇濆瓨銆傜湡鏈洪儴缃蹭娇鐢?step 20,000銆備笉瑕佹妸鏃х殑鈥滅粷瀵瑰叧鑺傚姩浣滄ā鍨嬧€濅笌鏈?config 鐨?delta 璁粌鍙樻崲娣风敤銆?
## 鍘熷 RL Token

RL Token 璁粌鍐荤粨 蟺0.5 鏉冮噸锛屽彧閲嶅缓 prefix embeddings锛?
```bash
uv run scripts/train_rlt_token_jax.py \
  --config-name pi05_piper_greenblock_5090_jax_delta_v1 \
  --checkpoint-dir /path/to/sft/20000 \
  --output-dir /path/to/rl_tokens/run_v1
```

璁粌瀹屾垚鍚庡鍑虹嚎涓婃墍闇€鐨?encoder-only checkpoint锛?
```bash
uv run scripts/export_rlt_token_encoder_only.py \
  --source /path/to/rl_tokens/run_v1/step_010000 \
  --output /path/to/rl_tokens/run_v1/step_010000_encoder_only
```

鍘熷缁撴瀯銆佹渶缁堣缁冩寚鏍囥€侀獙璇佸拰缂撳瓨娉ㄦ剰浜嬮」瀹屾暣璁板綍鍦?`docs/RL_TOKEN_AND_PHASE_CLASSIFIER.md`銆俙scripts/train_rlt_token_pytorch.py` 鏄棭鏈熷鐓у疄鐜帮紝涓嶆槸褰撳墠閮ㄧ讲鏉冮噸鐨勬潵婧愩€?
## ResNet 闃舵鍒嗙被鍣?
浣跨敤 `configs/phase_annotations_template_30eps.csv` 璁板綍姣忎釜 episode 鐨?`[phase_start_t, phase_end_t]`锛屽啀渚濇鎵ц锛?
```bash
uv run scripts/build_piper_phase_dataset_from_intervals.py --help
uv run scripts/train_piper_phase_classifier.py --help
uv run scripts/validate_piper_phase_classifier.py --help
uv run scripts/benchmark_phase_classifier_runtime.py --help
```

妯℃澘淇濈暀浜嗗綋鏃?30 鏉?warm-up 鏁版嵁鐨勭湡瀹?interval锛屼絾鏁版嵁鏍硅矾寰勬槸 5090 蹇収璺緞锛涜縼绉绘椂蹇呴』鏀规垚鑷繁鐨?parquet 璺緞銆傜姝㈡彁浜ゅ鍑虹殑璁粌鍥惧儚鍜?QA grid锛岄櫎闈炲凡纭鏃犳晱鎰熺幇鍦哄唴瀹广€?
## 绾帹鐞嗕笌鍥哄畾璇勪及

绾帹鐞嗕笉闇€瑕?Sense锛?
```bash
bash ~/piper_jax_inference_v1/start_jax_inference_mode.sh
EPISODES=20 bash ~/piper_jax_inference_v1/run_h50_inference.sh
```

鍥哄畾 Actor checkpoint 璇勪及鍚屾牱涓嶄娇鐢?Sense锛?
```bash
bash ~/piper_jax_inference_v1/run_fixed_checkpoint_eval_gripper_close_v3.sh \
  --step 821 --episodes 20
```

`run_h50_inference.sh` 姣忎釜 episode 鍏堝仛 4 绉掓煍鍜屽浣嶏紱鎵ц H50锛屽苟鍦ㄥ姩浣滆竟鐣屾彁鍓?4 姝ラ鍙栥€傛搷浣滆€呯敤 `1/0` 璁板綍缁撴灉銆傚紑濮嬪墠鑴氭湰浼氭嫆缁濅笌 Pika 鏁版嵁閲囬泦/鐩告満鑺傜偣鎴栧彟涓€ rollout 骞跺彂杩愯銆?
## 鍦ㄧ嚎 RLT

褰撳墠 fresh-zero lineage 鐨勪竴閿叆鍙ｏ細

```bash
bash ~/piper_jax_inference_v1/start_jax_rlt_mode.sh
```

濡傛灉娌℃湁澶嶅埗 5090 鐨?session state锛屽簲鍏堝垱寤烘柊鐨勩€佸畬鍏ㄧ嫭绔嬬殑 fresh-zero lineage锛岃€屼笉鏄吉閫?selector锛?
```bash
bash ~/gripper_close_v3_staging_20260727/run_greenblock_gripper_close_v3_fresh_warmup.sh \
  --name my_gripper_rlt_v1 \
  --warmup-episodes 40 \
  --state-dir .online_rlt_my_gripper_rlt_v1
```

瀹冮粯璁や娇鐢細

```text
lineage  = greenblock_rlt_fresh_v3_20260729
state    = .online_rlt_persistent_gripper_v3_fresh
warm-up  = 40 admitted episodes
H50/C10  = base horizon 50, residual chunk 10, stride 10, n-step 10
UTD      = 1
batch    = 256
```

episode 瀹屾垚鍚庣殑浜や簰璇箟锛?
- `t` / `y`锛氬噯鍏ュ綋鍓?episode锛屽苟杩愯鍦ㄧ嚎 A鈥揅 update銆?- `r` / `n`锛氫笉鍑嗗叆锛屼繚鐣欑浉鍚?Actor 閲嶈瘯銆?- `e` / `q`锛氫笉鍑嗗叆骞跺仠姝€?
reward 涓庘€滄槸鍚﹀噯鍏モ€濇槸涓や欢浜嬨€傜湡瀹炲け璐ユā寮忓彲浠ョ粰 reward 0 骞跺噯鍏ワ紝缁?Critic 鎻愪緵杈圭晫淇℃伅锛涚浉鏈哄紓甯搞€佸す鐖?鍦烘櫙鏈浣嶃€佽鎿嶄綔绛?out-of-distribution 浜嬫晠涓嶈鍑嗗叆銆備换鍔℃渶缁堝畬鎴愪絾缁忓巻浜屾鎶撳彇锛屽簲鎸夐鍏堝浐瀹氱殑璇勪及鍗忚鏍囨敞锛屼笉瑕佷复鏃舵敼鍙ｅ緞銆?
鍦ㄧ嚎缂撳瓨浼氬厛杩囨护涓嶈兘鏋勬垚涓ユ牸 C10/`t+10` 鐨勮锛屽啀閫氳繃涓€涓?WebSocket 鎵硅姹傞€愭牱鏈繚鎸?batch-size-one 鐨?frozen encoder 鏁板€艰矾寰勶紱manifest 涓庡唴瀹瑰搱甯屽尮閰嶆椂鍙閲忚绠楁柊 episode銆備笉鑳戒负浜嗛€熷害鍙栨秷鍚岃鍒?C10銆佺簿纭墽琛屾椂 `a_ref` 鎴?terminal 鍚堢害銆?
鎴嚦蹇収锛歠resh-zero lineage 宸茶缁?119 涓?admitted episodes銆?21 涓缁?transitions锛屽綋鍓?selector 涓?step 821锛屾渶鍚庝竴娆℃洿鏂版柊澧?5 steps銆傝鐘舵€佸彧浣滀负瀹¤璁板綍锛屼笉闅?Git 鍒嗗彂銆傝瑙?`artifacts/metadata/rlt_step_00000821.json`銆?
## 娴嬭瘯

```bash
uv run pytest tests/test_rlt_phase_gate.py src/openpi/rlt/real/phase_classifier_test.py
uv run pytest tests/test_pure_inference_wrappers.py tests/test_policy_hardware_rollout.py
uv run pytest scripts/piper_rlt
```

shell 闈欐€佹鏌ワ細

```bash
find . -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

鐪熸満娴嬭瘯椤哄簭蹇呴』鏄細鍙鏋氫妇 鈫?dry-run/鍗忚 smoke 鈫?policy server 甯搁┗妫€鏌?鈫?CAN 鍙鐘舵€?鈫?浣庡箙搴﹀崟姝ュ姩浣?鈫?鍗?episode 鈫?澶?episode銆備笉瑕佷粠鈥滄祴璇曢€氳繃鈥濈洿鎺ユ帹瀵间负鈥滃彲浠ユ棤浜哄€煎畧鈥濄€?
## 宸茬煡闄愬埗

- 2026-08 鏇存崲鑵曢儴 D405 鍚庤瀵熷埌鏄庢樉杩囨洕/鍩熷亸绉伙細鏃у浘鍧囧€肩害 120銆侀ケ鍜屽儚绱犺繎 0锛涙柊寮傚父 episode 鍧囧€肩害 204鈥?14銆侀ケ鍜屽儚绱犵害 32%鈥?6%銆傝繖浼氳鐧借壊鐩爣娑堝け銆佅€0.5 灏忓箙鎶栧姩锛屽苟鎶?phase 姒傜巼鍘嬪埌绾?0.008锛屽鑷?Actor 涓嶈繘鍏ャ€傚厛淇浉鏈烘洕鍏変笌鍩熶竴鑷存€э紝鍐嶇户缁噯鍏ユ暟鎹€?- ResNet 楠岃瘉闆嗘潵鑷棫鐩告満鍒嗗竷锛?7.3% held-out accuracy 涓嶄唬琛ㄦ洿鎹㈢浉鏈哄悗鐨勫彲闈犳€с€?- systemd 鏂囦欢浣嶄簬 `deploy/systemd/5090-snapshot/`锛屾槸鎴愬姛涓绘満鐨勭簿纭矾寰勫揩鐓э紝涓嶆槸閫氱敤瀹夎鍣ㄣ€?- 浠撳簱涓嶅寘鍚?RLT 璁烘枃銆佺涓夋柟 SDK 婧愮爜銆佹ā鍨嬫潈閲嶆垨鐪熷疄鏁版嵁銆?
## 璁稿彲璇佷笌鏉ユ簮

鏈粨搴撳熀浜?Physical Intelligence OpenPI 浠ｇ爜骞朵繚鐣?Apache-2.0 涓?Gemma 鐩稿叧璁稿彲璇侊紱Piper SDK銆丳ika ROS 鍜?LeRobot 鎸夊悇鑷笂娓歌鍙崟鐙幏鍙栥€傝嚜瀹氫箟 Piper/RLT 鏂囦欢鐨勬潵婧愬拰蹇収杈圭晫瑙?`docs/SOURCE_PROVENANCE.md`銆?
