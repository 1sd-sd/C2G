# C2G 参数高尔夫 —— 提交成果包

**挑战：** C2G Parameter Golf · 极限约束下的语言模型训练
**作者：** lishengdan
**目标等级：** Level 2（主）/ Level 3（条件达成）/ Level 4（仅形式交付）
**提交日期：** 2026-09-29

---

## ⚠️ 先读这一段：这个提交是什么，不是什么

**是什么：** 一份**完整可运行**的参数高尔夫训练系统 —— 设备无关的训练脚本、全组件环境变量化的消融台、自建确定性代理语料流水线、可切换的量化导出（位宽 int6/int8、裁剪规则 percentile/SDClip、算法 RTN/GPTQ，出货用的是 **int6 + percentile + RTN**）、独立复现评估器，以及在这些之上跑出来的 **34 次真实实验**（15 个实验臂 × 2–3 seeds，配对比较）。

**不是什么：** **不是一次官方榜单提交。** 本执行环境：

| 项 | 实际情况 |
|---|---|
| GPU | **无**（`nvidia-smi: command not found`） |
| huggingface.co / hf-mirror.com | **HTTP 000，不可达**，因此拿不到 FineWeb |
| 可用算力 | Intel 22 逻辑核心 CPU |

所以本地测得的 BPB **不是** FineWeb 上的官方 BPB，本提交**不声称**达到 1.18 / 1.12 / 1.085 任何一条线。所有引用 `1.2244` / `1.1586` / `1.0810` 的地方都标注为**官方公开数字**。

**这份提交真正提供的价值：** 在**同代码、同数据、配对 seed** 的严格条件下，给出各组件在"部署后指标"上的**排序证据**，并把「量化税」从一个模糊的实现细节变成一个被测量的量。详见 `lishengdan_C2G_方案设计.md` §5.3 与 `lishengdan_C2G_AAR.md`。

---

## 目录

| 文件 | 说明 |
|---|---|
| `README.md` | 本文件：总览、安装、一键复现 |
| `lishengdan_C2G_方案草案.md` | **算力申请前置材料**（≥500 字，回答 4 个门槛问题）；含 8 条预注册预测 |
| `lishengdan_C2G_方案设计.md` | 目标等级、技术选型、实验矩阵、风险登记表 |
| `lishengdan_C2G_train_gpt.py` | **最终训练脚本**（设备无关，全组件环境变量开关） |
| `lishengdan_C2G_eval_bpb.py` | 独立复现评估器：从 artifact 重算 BPB，不一致即报错退出 |
| `lishengdan_C2G_submission.json` | 提交元数据（模型形状、量化方案、字节核算、3 seeds 结果） |
| `lishengdan_C2G_submission.tar.gz` | **artifact**（代码 + 量化权重 + tokenizer），已校验 ≤ 16,000,000 B |
| `lishengdan_C2G_logs/` | 全部实验的完整训练日志（每个 seed 一份） |
| `lishengdan_C2G_ablation.md` | 消融实验报告：每个组件的单独贡献 + 组合效应 |
| `lishengdan_C2G_leaderboard.md` | BPB 对比表：本方案 vs baseline vs 当前 SOTA（含来源标注） |
| `experiments/results/prediction_scorecard.md` | **8 条预注册预测的记分卡**（自动生成），区分「不成立」与「功效不足、不可判定」 |
| `experiments/results/power.json` | 本装置的分辨率地板（配对差的 95% t 区间半宽），供上述判定使用 |
| `lishengdan_C2G_AI日志.md` | **开发全过程 AI 使用记录**（含被否决的 AI 建议与 AI 出错的记录） |
| `lishengdan_C2G_拿来说明.md` | 从 nanoGPT / 官方 repo / 历史 top 方案拿了什么、改了什么、为什么改 |
| `lishengdan_C2G_AAR.md` | **复盘**：预注册预测逐条对账、失败分析、改进方案 |
| `lishengdan_C2G_tech_report.pdf` | 英文技术报告（NeurIPS workshop 双栏格式） |
| `experiments/` | 全部脚本：语料构造、tokenizer 训练、实验驱动、图、报告、打包 |
| `artifact/` | 导出的量化权重与 artifact |

---

## 一键复现

### 0. 环境（Windows / 无 GPU 也能跑）

```bash
# 全部依赖装在 D 盘隔离 venv，不污染系统环境（路径可改，下面统一用 $VENV）
VENV=D:/c2g_env
python -m venv "$VENV"
"$VENV/Scripts/python.exe" -m pip install torch \
    --index-url https://download.pytorch.org/whl/cpu
"$VENV/Scripts/python.exe" -m pip install numpy tqdm sentencepiece brotli matplotlib reportlab \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

> 清华源（`pypi.tuna.tsinghua.edu.cn`）在本环境取不到 torch 的 wheel 列表
> （`Could not find a version that satisfies the requirement torch`），换阿里源 +
> PyTorch CPU index 才通。这是实测记录，见 `lishengdan_C2G_AI日志.md`。
>
> 装完后，下面所有命令里的 `$PY` 都可以用 `C2G_PYTHON` 环境变量指定解释器；
> `experiments/wait_and_finalize.sh` 会自动探测（优先 `$C2G_PYTHON`，
> 其次 `/d/c2g_env`，最后 `python3`/`python`，并要求能 `import torch`）。

### 1. 重建代理语料（确定性，约 4 分钟）

```bash
cd experiments/corpus
python fetch_proxy_corpus.py --target-bytes 40000000 --out raw/simplewiki.txt
python prepare_corpus.py --text raw/simplewiki.txt --out-root . --vocab-sizes 1024 4096 8192
```

产出：`tokenizers/fineweb_{1024,4096,8192}_bpe.model` 与官方**字节级一致**的
`.bin` 分片（256×int32 头 + uint16 token）。

### 2. 跑一次训练

**实际提交的是 `A2_sp8192`**（只换词表，其余全部默认）。这不是偏好，是
`package_submission.py` 的 `pick_best_arm()` 按实测 BPB 自动选的：全栈臂反而更差
（详见 `lishengdan_C2G_AAR.md` §3.1.2）。下面第一段复现**出货模型**，
第二段复现**全栈臂**用于对照。

```bash
# 在提交根目录下执行：所有路径都是相对路径（见下方"sentencepiece 路径注意"）
# 可移植写法：PY=${C2G_PYTHON:-python}
PY=${C2G_PYTHON:-python}

# --- (a) 出货模型 A2_sp8192：词表 8192，其余保持默认 ---
OUT_DIR=artifact DEVICE=cpu COMPILE=0 MAX_WALLCLOCK_SECONDS=0 \
  ITERATIONS=400 VAL_LOSS_EVERY=200 TRAIN_BATCH_TOKENS=4096 TRAIN_SEQ_LEN=256 \
  NUM_LAYERS=6 MODEL_DIM=128 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 \
  VOCAB_SIZE=8192 \
  DATA_PATH=experiments/corpus/datasets/proxy_sp8192 \
  TOKENIZER_PATH=experiments/corpus/tokenizers/fineweb_8192_bpe.model \
  SEED=2024 RUN_ID=reproduce_A2_seed2024 RESULT_JSON=/tmp/reproduce_A2.json \
  "$PY" lishengdan_C2G_train_gpt.py

# --- (b) 全栈臂 B2_combo_qaema（对照，BN 结果比 (a) 差约 0.10 BPB）---
OUT_DIR=artifact_fullstack DEVICE=cpu COMPILE=0 MAX_WALLCLOCK_SECONDS=0 \
  ITERATIONS=400 VAL_LOSS_EVERY=200 TRAIN_BATCH_TOKENS=4096 TRAIN_SEQ_LEN=256 \
  NUM_LAYERS=6 MODEL_DIM=128 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 \
  VOCAB_SIZE=8192 \
  DATA_PATH=experiments/corpus/datasets/proxy_sp8192 \
  TOKENIZER_PATH=experiments/corpus/tokenizers/fineweb_8192_bpe.model \
  QK_GAIN_INIT=5.25 PARALLEL_RESID_FROM=4 LAYER_SCHEDULE=0,1,2,3,4,5,2,3,4,5 \
  EMA_DECAY=0.996 EMA_START_FRAC=0.65 EMA_QUANT_AWARE=1 \
  QAT_START_FRAC=-1 QAT_BITS=6 \
  QUANT_BITS=6 QUANT_CLIP=sdclip QUANT_METHOD=gptq COMPRESS=brotli \
  SEED=2024 RUN_ID=reproduce_B2_seed2024 RESULT_JSON=/tmp/reproduce_B2.json \
  "$PY" lishengdan_C2G_train_gpt.py
```

> 未列出的环境变量一律取训练脚本的默认值。**注意默认值不是脚手架的导出规则**：
> `QUANT_BITS` 默认 **6**（不是 8）、`QUANT_CLIP` 默认 **percentile**。
> 这一点在早期被写错过，见 `lishengdan_C2G_AAR.md` §4.7。

### 3. 验证 artifact 可复现分数

```bash
python lishengdan_C2G_eval_bpb.py \
  --artifact artifact/final_model.quant.bin \
  --submission lishengdan_C2G_submission.json \
  --data-path experiments/corpus/datasets/proxy_sp8192 \
  --tokenizer experiments/corpus/tokenizers/fineweb_8192_bpe.model
```

评估器会从磁盘上的 artifact 重建模型、反量化、重算 BPB，**如果与
`submission.json` 的声明相差超过 1e-6 就直接报错退出**。

> 已实测：对**实际出货的 artifact**（`A2_sp8192_seed2024`）独立重算得到
> `val_bpb = 1.854895`，与 manifest 声明相差 **3.3e-07**，远在 1e-6 合同之内。

### 4. 复现整张消融表

```bash
cd experiments
bash wait_and_finalize.sh          # 矩阵 + 统计 + 图 + 报告 + 打包，一条龙
# 或者逐步来：
python run_matrix.py --workers 4 --threads-per-worker 5   # 34 次运行，约 1.5 小时
python run_matrix.py --summarize-only
python analyze_results.py          # 生成记分卡与分辨率地板
python write_ablation_report.py
python make_figures.py
python make_report.py
python package_submission.py
```

---

## ⚠️ sentencepiece 路径注意（Windows）

`sentencepiece` 通过 C++ `fopen()` 打开模型文件，在 Windows 上对**含非 ASCII 字符的绝对路径**会失败（本提交的中文父目录正好命中）。因此：

- 训练脚本的 `DATA_PATH` / `TOKENIZER_PATH` 请用**相对于提交根目录的路径**；
- 所有实验脚本已按此约定写好（`run_matrix.py` 里用相对路径 + `cwd=提交根目录`）。

这是一个真实的踩坑记录，写在 `lishengdan_C2G_AAR.md` 里。

---

## 结果速览（本地代理语料，非官方成绩）

> 完整表格见 `lishengdan_C2G_ablation.md`；官方榜单对比见 `lishengdan_C2G_leaderboard.md`。

- **配置：** 6 层 × 128 维、8 头 / 4 KV 头、MLP 2×、seq_len 256、4096 token/步、400 步（1.64M token）
- **指标：** `final val_bpb` = 量化 + 压缩 + roundtrip 之后的 BPB（这才是"成绩单"上的数）
- **方法：** 15 臂 × 2–3 seeds，**同 seed 配对比较**（`Δ = bpb(arm) − bpb(baseline)`）

**三条最重要的结论**（每一条都有实测支撑，也都有它的局限）：

| # | 结论 | 数字 | 局限 |
|---|---|---|---|
| 1 | **换大词表是唯一站得住的改进** | SP8192 相对 SP1024：**−0.2248** BPB（阈值 −0.02） | 该臂同时多看了 54% 字节，两者分不开（§5.3） |
| 2 | **堆组件反而更差** | 全栈 `B1_combo` **1.9538** vs 只换词表 `A2_sp8192` **1.8569**，差 **+0.0969** | 小模型尺度，外推到 11L×512d 未验证 |
| 3 | **本提交的原创组件被自己的实验证伪** | QA-EMA（P7）Δ = **+0.0096 ± 0.0008**，判定 `NOT SUPPORTED` | 无——这个区间很窄，是真的更差 |

**因此实际提交的是 `A2_sp8192`**（只换词表，其余默认），由 `package_submission.py`
的 `pick_best_arm()` 按实测 BPB 自动选定，不是人工偏好。出货模型：
`val_bpb = 1.854895`，artifact **1,736,896 B**（限额 16,000,000，余量 89%）。

（表格由 `experiments/run_matrix.py` 生成，数字全部来自 `experiments/results/summary.json`。）

---

## 官方赛道配置（如何把它搬到 8×H100）

> ⚠️ **这一段要带着上一节的结论读。** 下面的配置是本提交**设计时**推导出来的，
> 但本地筛选的结果**并不支持它**：QK-Gain、并行残差、深度递归、EMA 在小尺度上
> 都没测出正收益（`A6_ema` 甚至明确变差），SDClip 反而增大量化税。
> 也就是说，**照抄下面这段很可能不如只换词表**。我保留它是因为它记录了原设计意图，
> 而不是因为它被验证过——它没有被验证过。

本脚本的设备无关设计意味着同一份文件可以直接切到官方配置：

```bash
SEED=42 DATA_PATH=./data/datasets/fineweb10B_sp8192 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model VOCAB_SIZE=8192 \
  NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4 \
  TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024 MAX_WALLCLOCK_SECONDS=600 \
  QK_GAIN_INIT=5.25 PARALLEL_RESID_FROM=7 \
  LAYER_SCHEDULE=0,1,2,3,4,5,3,4,5,3,4,5,6,7,8,9,10 \
  EMA_DECAY=0.9965 EMA_START_FRAC=0.28 EMA_QUANT_AWARE=1 WEIGHT_DECAY=0.095 \
  QAT_START_FRAC=-1 QAT_BITS=6 \
  QUANT_BITS=6 QUANT_EMBED_BITS=8 QUANT_CLIP=sdclip SDCLIP_K=12.85 \
  QUANT_METHOD=gptq COMPRESS=brotli \
  torchrun --standalone --nproc_per_node=8 lishengdan_C2G_train_gpt.py
```

**上表是按本提交的组件设计推导出的配置，我没有 8×H100 环境去验证它。** 我会在
`lishengdan_C2G_AAR.md` 里把这一点作为首要待办列出，而不是让它看起来像已验证过。
