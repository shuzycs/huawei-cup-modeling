# 问题 2：局部模态缺失下的情感预测实验手册

本手册供在 **Linux 服务器**上工作的 AI Agent 从零执行。目标是依据赛题提供的 CMU-MOSEI 数据，训练一个在文本、语音、视觉局部连续缺失时仍能输出**三分类情感极性**和 **[-3, 3] 连续情感强度**的模型；完成验证、消融、规律分析，并生成附件 3 的全量预测文件。下文给出可执行的默认方案。所有未注明“已核实”的模型结构和超参数均为**待验证的实验设计**，不得写成已取得的结果。

## 0. 已核实事实、边界和最终交付

### 0.1 数据事实

- 使用仓库根目录下 `data/extracted/E题数据/附件2-数据集特征文件/` 的 `aligned_50.pkl` 与 `label.xlsx`。对齐版分为 `train=3395`、`valid=728`、`test=727`，合计 4850 条。标签编码为 `0=Negative`、`1=Neutral`、`2=Positive`；连续标签范围为 `[-3,3]`。
- 附件 2 对齐版有 `text_bert (N,3,50)`、`audio (N,50,74)`、`vision (N,50,35)`，也有预计算的 `text (N,50,768)`。**附件 3 对齐版没有 `text`**；其 30 个文件各包含 `test` 字典，字段是 `text_bert (1,3,50)`、`audio (1,50,74)`、`vision (1,50,35)`。因此最终模型的文本输入必须来自 `text_bert`，不能只在附件 2 的 `text` 上训练。
- `text_bert` 的三路依次为 token IDs、attention mask、token type IDs。读取后验证整数性并转为 `int64`。单条视频切片是一个样本；50 是样本内部的最多 50 个序列位置。
- 对齐版 `.pkl` 实际没有 `annotations` 字段；文字极性在 `label.xlsx` 的 `annotation` 列。
- 赛题要求：只在附件 2 `train` 学习模型参数，用 `valid` 选择结构、超参数和阈值；附件 3 是无标签专项测试集，只用于最终推理。不得使用其他情感数据集训练、微调或调参。开源预训练模型可用，但要记录名称、版本、来源和用途。提交附件总量不超过 50 MB。

### 0.2 工作原则

1. 默认只做**对齐版**，训练和附件 3 推理不混用对齐与未对齐文件。未对齐版需另起实验，不能把两版指标直接混为同一结果。
2. 可读附件 3 的文件名、字段、形状以实现接口；**不得根据附件 3 的特征分布或预测结果选择超参数**。
3. 附件 2 的 `test` 可在模型与配置完全冻结后做一次独立内部检验，不参与训练、早停或选择。
4. 论文中的任何指标必须来自实际日志；没有运行的实验标为“未完成”，不得填估计值。

### 0.3 验收交付物

在仓库中实现 `src/problem2/`、`configs/problem2.yaml`、`scripts/run_problem2.sh` 和简洁运行说明。在 `outputs/problem2/` 生成：环境清单、数据核查报告、各实验的配置与日志、最佳推理权重、验证集逐样本预测、指标表、消融表、图、错误分析、附件 3 的 30 行预测 CSV、复现报告。最终提交包仅含赛题要求的代码、配置、说明、轻量权重和 CSV；**不打包原始数据、虚拟环境、缓存、优化器状态和大日志**。

## 1. 服务器：从虚拟环境开始

以下命令以 Bash 和仓库根目录为前提。Agent 先确认 `pwd` 指向该仓库；如果服务器目录不同，保留仓库内相对路径即可。不要把 Windows 本机的 `D:` 路径写进服务器代码。

```bash
pwd
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python --version
```

若 `python3.11 -m venv` 失败，先在服务器安装 Python 3.11 及其 `venv` 组件，再重试；不要改用系统 Python 直接安装。把 `.venv/` 加入 `.gitignore`。Python `venv` 的创建方式见[官方文档](https://docs.python.org/3.11/tutorial/venv.html)。

随后记录服务器条件：

```bash
uname -a
free -h
df -h .
nvidia-smi
```

`nvidia-smi` 不存在时记录为 CPU 环境，不视为数据错误。对齐版 `.pkl` 约 0.93 GiB，首次反序列化可能需要数 GiB 内存；建议服务器有 **至少 16 GiB RAM**、足够的磁盘缓存空间。GPU 推荐但非硬性要求；若有 NVIDIA GPU，先按[PyTorch 官方安装选择器](https://docs.pytorch.org/get-started/locally/)选择与驱动兼容的 Linux/Pip/CUDA 命令。将**实际执行的安装命令**写入 `outputs/problem2/env/install_commands.txt`。仅 CPU 时选择官方 CPU wheel。不要猜测服务器 CUDA 版本。

安装其余依赖并冻结实际版本：

```bash
python -m pip install "numpy>=1.26,<3" "pandas>=2.2,<3" "openpyxl>=3.1,<4" \
  "scikit-learn>=1.4,<2" "transformers>=4.40,<5" "huggingface_hub>=0.26,<2" \
  "safetensors>=0.4,<1" "matplotlib>=3.8,<4" "PyYAML>=6,<7" \
  "tqdm>=4.66,<5" "pytest>=8,<10"
mkdir -p outputs/problem2/env
python -m pip freeze > outputs/problem2/env/pip_freeze.txt
python -c "import torch,transformers,numpy; print(torch.__version__, transformers.__version__, numpy.__version__, torch.cuda.is_available())"
```

上面的依赖命令应在 **PyTorch 安装成功后**执行。若服务器不能联网，改用服务器提供的 wheel 镜像和预先放置的模型文件；保留同样的版本记录。不要在缺少依赖时悄悄切换到另一种文本模型。

### 1.1 预训练文本编码器

默认使用 [`google/bert_uncased_L-2_H-128_A-2`](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2)：这是约 4.43M 参数的通用小型 BERT，仓库的 `model.safetensors` 约 17.7 MB，便于控制最终附件体积。它**不是情感数据微调模型**。首次下载时固定 Hugging Face commit SHA，记录模型文件 SHA256；离线服务器需提前把同一 revision 的模型文件放入服务器。模型目录只保留用于推理的 `config.json`、`model.safetensors`、`vocab.txt` 等必要文件，避免把同一权重的 `.bin` 与 `.safetensors` 都放入提交包。

Agent 必须做词表兼容检查：在附件 2 取至少 20 条 `raw_text`/`text_bert`，确认特殊 token ID、词表大小，并把若干 `text_bert` ID 反解为词元，与转写文本人工核对。再用该模型 tokenizer 编码并记录一致情况；原数据可能做过标点、截断等预处理，因此**不以逐 ID 完全一致作为唯一条件**。若反解词元与原文大面积不符，停止使用该模型并找出匹配词表，不可仅因维度相同就继续。训练、验证、附件 3 均使用完全相同的编码器权重。默认冻结 BERT 参数，第一轮只训练后续时序编码与预测头。若验证效果不足，再把解冻后两层作为单独实验，而不是偷偷改动基线。

### 1.2 Agent 应创建的目录与配置

源码至少包含 `inspect_data.py`、`prepare_data.py`、`dataset.py`、`masks.py`、`model.py`、`train.py`、`evaluate.py`、`analyze.py`、`infer.py`，均放在 `src/problem2/`，并创建包所需的 `__init__.py`。配置文件 `configs/problem2.yaml` 至少写明以下键；路径始终相对仓库根目录，模型 revision 必须填**已核实的完整 commit SHA**：

```yaml
data:
  aligned_pkl: data/extracted/E题数据/附件2-数据集特征文件/aligned_50.pkl
  labels_xlsx: data/extracted/E题数据/附件2-数据集特征文件/label.xlsx
  challenge_dir: data/extracted/E题数据/附件3-模态缺失特征样本/对齐版本
  cache_dir: outputs/problem2/cache
text_model:
  repo_id: google/bert_uncased_L-2_H-128_A-2
  revision: REPLACE_WITH_VERIFIED_COMMIT_SHA
  frozen: true
model:
  hidden_dim: 128
  dropout: 0.2
training:
  seeds: [42, 52, 62]
  batch_size: 32
  max_epochs: 40
  patience: 8
  learning_rate: 0.001
  weight_decay: 0.01
  regression_weight: 0.5
  clean_probability: 0.25
  mask_rates: [0.10, 0.30, 0.50, 0.70]
output_dir: outputs/problem2
```

在启动训练前检查配置无占位符、模型 revision 已固定、所有数据路径存在。`outputs/problem2/` 与 `.venv/` 均不纳入 Git 或最终提交包。

## 2. 数据核查与转换

### 2.1 文件定位和基本核查

实现 `python -m src.problem2.inspect_data --config configs/problem2.yaml`。检查下列条件，失败即停止训练并输出明确错误：

1. `aligned_50.pkl`、`label.xlsx`、附件 3 对齐版 30 个 `.pkl` 都存在；只反序列化**可信的赛题文件**，因为 Pickle 可执行代码。
2. 附件 2 顶层为 `train/valid/test`，各字段第一维分别是 3395/728/727；`text_bert/audio/vision` 的后两维分别是 `(3,50)/(50,74)/(50,35)`，标签长度和 `id` 长度一致。
3. `classification_labels` 仅为 0/1/2，`regression_labels` 有限且在 `[-3,3]`；用 `id` 与 Excel 的 `video_id + '$_$' + clip_id` 连接并核对两种标签（浮点强度用合理容差），禁止只按 Excel 行号盲目连接。
4. 三个划分的样本 ID 不重复；数据、标签无 NaN/Inf。记录各字段 dtype、最小最大值、非零行数分布、标签分布、单个文件大小。
5. 附件 3 每个文件仅作 schema 检查：`test` 内三字段形状为 `(1,3,50)/(1,50,74)/(1,50,35)`；结果顺序按文件名 `01` 到 `30` 明确排序。

核查结果保存为 `outputs/problem2/data_audit.json` 和 `data_audit.md`。第一次读取大 `.pkl` 后，将每个划分所需数组转为独立 `.npy` 或 `.npz` 缓存；后续训练直接读缓存，避免每次加载近 1 GiB Pickle。缓存放 `outputs/problem2/cache/`，不提交。保留原始 `id` 顺序与标签映射。推荐语音、视觉存 `float32`；文本 IDs/掩码存整数类型。记录原文件 SHA256 和转换脚本版本。

### 2.2 掩码定义与预处理顺序

对每个样本、模态 $m\in\{T,A,V\}$、位置 $t\in\{1,\ldots,50\}$，定义：

- $v_{m,t}=1$：原始有效位置；填充位置为 0。
- $o_{m,t}=1$：本次输入中实际可观测；填充或局部缺失为 0。
- 局部缺失为 $v_{m,t}=1,o_{m,t}=0$，与原本填充 $v_{m,t}=0$ 分开记录。

在完整附件 2 中，文本有效性由 `text_bert` 的 attention mask 确定；**内容可观测性另须排除 `[CLS]`、`[SEP]`**，这两个特殊 token 不能让内容完全缺失的文本模态被误判为“仍有文本”。模拟文本缺失时也不遮蔽特殊 token。对语音、视觉，先在**未标准化**的完整训练样本中检测非零行和尾部填充；核查其与文本长度的关系，并在数据报告中给出规则。附件 3 只有缺失后的零行，不能完美反推原始有效长度；推理时把全零行当“不可观测”并在模型中排除。模型不能依赖推理时不可获得的“真实缺失区间标签”。

只在训练集可观测的语音、视觉行上拟合每维均值、标准差；方差接近零的维度用安全下界。先根据**原始值**提取可观测掩码，再对可观测行标准化，最后将不可观测行置零。验证、附件 2 `test`、附件 3 都使用**训练集**统计量；保存 `normalizer.npz`。训练时给文本遮蔽必须在进入 BERT **之前**修改 token IDs 和 attention mask；不能先用完整文本过 BERT、再遮盖输出，因为剩余 token 的上下文表示已包含被遮蔽内容。

### 2.3 连续缺失生成器

实现确定性的 `SpanMasker(seed, epoch, sample_id)`，不要用 Python 内置 `hash()` 生成种子。默认初始配置：每批约 25% 样本保留完整输入；其余样本在 1 个模态上遮蔽的概率 70%，在 2 个模态上遮蔽的概率 30%。对每个被选模态，从**原始有效内容位置**选一段连续区间，长度比例从 `{0.10,0.30,0.50,0.70}` 均匀采样；起点覆盖开头、中部、结尾。短样本无法实现某一长度时，记录实际比例并保持至少一个内容位置可观测。此配置是实验起点，不是赛题给定的真实缺失分布。

- 文本：在 token 序列的内容位置将区间的 token ID 设为 `[PAD]`，attention mask 设为 0，token type ID 设为 0；保留特殊 token。进入 BERT 后再次把不可观测位置的输出乘零。
- 语音/视觉：将该区间整行特征设零，同时更新 $o_{m,t}$。若标准化已经完成，必须**在标准化之后再次置零**。
- 单元检查：同一原始样本生成多个视图时标签不变；被遮蔽区间连续；未选模态的特征不变；只遮蔽有效位置；掩码与零行一致；缺失处不能通过原始 `raw_text` 或附件 2 的预计算 `text` 绕过。

验证集的缺失图谱固定为：模态 `T/A/V` × 位置 `起/中/末` × 比例 `10/30/50/70%`，加上三模态完整组。每个条件至少固定 3 个随机视图，并对**所有模型使用同一批视图**；另设少量双模态缺失组。将完整视图和所有缺失视图的索引、种子保存为 `validation_masks.json`，训练中不得重新抽样验证视图。

## 3. 模型定义

实现 `src/problem2/model.py` 中统一的模型接口：输入 `text_bert, audio, vision, observed_masks`，输出 `class_logits:[B,3]`、`intensity:[B]`、`fusion_weights:[B,3]`。四个主实验共用同一文本编码器与数据预处理，只有明确列出的机制不同。

### 3.1 模态编码与汇总

1. 冻结的小型 BERT 把 `text_bert[:,0,:]`（IDs）、`[:,1,:]`（attention mask）、`[:,2,:]`（token type IDs）变成文本时序表示；根据实际 `hidden_size` 投影到公共隐藏维 $h=128$。输入 IDs 必须为 `int64`。**不得把附件 2 的 `text` 当训练输入**。
2. 语音 `74→128`、视觉 `35→128` 分别用线性层 + LayerNorm + 1 层轻量时序编码器（例如核大小 3 的残差 1D 卷积）处理；输入附带观测标记。每层后显式屏蔽不可观测行，避免填充参与汇总。
3. 每个模态用**掩码注意力池化**得到样本向量 $u_m\in\mathbb R^{128}$：不可观测位置的注意力权重强制为 0；文本池化只用实际内容 token，不用 `[CLS]`、`[SEP]` 充当缺失后的内容。若某模态没有任何可观测内容，返回全零向量和可用标志 0，禁止 softmax 对空集合产生 NaN。

这里先做**模态内时序建模、样本级跨模态融合**，不要求 BERT 词元位置与音视频槽位逐个精确相同。这样既利用 50 位时序信息，也避免未经验证的逐位置硬对齐假设。

### 双层交叉注意力扩展的重训结果

独立配置 configs/problem2_cross_attention.yaml 将 E3/E4/E5 的门控融合替换为两层文本查询音视频的交叉注意力，再由可学习融合 token 汇总全部可观测位置。文本完全缺失时，该 token 仍可读取音视频；空 key 集合单独处理以避免 NaN。三种子同配置对照后，E5 的 36 缺失条件 macro-F1 从 0.5583 降到 0.5256，独立 test 强度 MAE 从 0.7489 升到 0.7861，因此默认仍采用下文的门控模型。逐种子与逐条件结果见 outputs/problem2_cross_attention/comparison_with_gate.md。

### 3.2 动态融合与双任务输出

对每个模态计算可观测统计量 $q_m$，例如可观测行数、是否至少有一行可用；训练和推理都必须**只从当前输入**按同一规则计算，不得把训练时已知、测试时未知的原始完整长度或真实缺失位置喂给门控网络。门控网络以 `[u_m,q_m]` 为输入产生 logit，对完全无可用内容的模态设极小 logit，再计算

\[
\alpha_m=\operatorname{softmax}_m(g_m),\qquad
z=\sum_{m\in\{T,A,V\}}\alpha_m u_m.
\]

记录每条样本的三模态权重，检查其非负且和为 1。分类头输出 3 个 logit；回归头输出 `3*tanh(raw_score)`，将强度限制到 `[-3,3]`。训练损失为

\[
\mathcal L=\mathcal L_{\mathrm{CE}}(\hat c,c)+\lambda\,\mathcal L_{\mathrm{Huber}}(\hat y,y).
\]

从 `λ=0.5` 起步，只允许在 `valid` 上从 `{0.25,0.5,1.0}` 中选择。分类头输出 `argmax` 后的 0/1/2，**不要仅靠回归分数的正负判中性**。

### 3.3 可选教师学生扩展

若主模型已跑通且验证有余量，新增 E4：用附件 2 `train` 的**完整输入**训练教师，再以同一 `train` 的连续缺失视图训练学生，在真实标签损失外增加教师软分类分布的 KL 损失和强度预测一致性损失。这借鉴 [CMAD](https://openaccess.thecvf.com/content/ICCV2025/html/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.html)，但不预设它必然提升本题。教师可以使用附件 2 的 `text` 作训练阶段辅助，**最终推理只用能读取 `text_bert/audio/vision` 的学生**。若 E4 未同时改善主要鲁棒指标和完整输入表现，就不纳入最终模型。

## 4. 实验矩阵与训练执行

### 4.1 逐层消融，保证比较可解释

| 编号 | 实验 | 目的 |
|---|---|---|
| E0 | 文本单模态，完整输入训练 | 检查文本基线与类别/强度任务能否学习 |
| E1 | 三模态按固定 50 位简单平均池化 + 拼接 MLP，仅完整输入训练 | 普通融合基线；零行也参与固定长度平均 |
| E2 | E1 + 连续区间缺失训练，仍按固定 50 位平均 | 单独检验缺失增强 |
| E3 | E2 + 显式掩码池化 + 动态模态门控 | 主模型；检验可用性建模与动态融合 |
| E4（可选） | E3 + 完整输入教师蒸馏 | 检验蒸馏增益 |

E0～E3 使用相同训练/验证划分、同一文本编码器、同一验证缺失视图。E1/E2 的差异只在训练遮蔽；E2/E3 的差异是掩码感知与门控，若需进一步识别两者贡献，再增加“仅掩码”或“仅门控”的补充实验。

默认训练配置：随机种子 `42/52/62` 三次独立训练；batch size 32；AdamW；主网络学习率 `1e-3`，权重衰减 `1e-2`；dropout 0.2；最多 40 epoch；验证无改进 8 epoch 早停；梯度范数裁剪 1.0。冻结 BERT 时 `eval()` + `no_grad()`，但**文本缺失必须先应用到输入 IDs/attention mask**。这些是起始值，若要修改只在验证集按预先记录的小范围搜索。每次训练保存 seed、配置副本、最佳 epoch、最佳权重、逐 epoch 指标和耗时。

实现以下可运行命令接口；Agent 在代码中兑现参数含义，并在 README 给出真实运行示例：

```bash
python -m src.problem2.inspect_data --config configs/problem2.yaml
python -m src.problem2.prepare_data --config configs/problem2.yaml
python -m src.problem2.train --config configs/problem2.yaml --experiment E0 --seed 42
python -m src.problem2.train --config configs/problem2.yaml --experiment E1 --seed 42
python -m src.problem2.train --config configs/problem2.yaml --experiment E2 --seed 42
python -m src.problem2.train --config configs/problem2.yaml --experiment E3 --seed 42
python -m src.problem2.evaluate --config configs/problem2.yaml --split valid --all-experiments
python -m src.problem2.analyze --config configs/problem2.yaml
```

上面的 `seed 42` 是单次冒烟和首个正式运行示例。正式 E0～E3 的三种子矩阵可由 `scripts/run_problem2.sh` 按 `experiment∈{E0,E1,E2,E3}`、`seed∈{42,52,62}` 循环调用；每项独立保存目录和配置，不覆盖上一项。

先用小批量和 1 epoch 做端到端冒烟运行：前向无 NaN，反向有梯度，单条样本形状正确，能保存并重载权重；确认输出文件结构后再启动正式三种子实验。若 GPU 显存不足，先减 batch size 并用梯度累积保持有效 batch，不改数据划分。若任一基线在几十个样本上无法过拟合，先检查字段顺序、标签映射、掩码和损失，暂停大规模训练。

### 4.2 模型选择规则

在运行前固定规则，避免事后挑结果：以 36 个单模态缺失条件的**平均 macro-F1 最大**为主要准则；MAE 较低者作为并列时的优先项；同时要求完整输入 macro-F1 相比 E1 不下降超过 2 个百分点，若有下降则列出性能折衷并选择通过约束的模型。对三种子取均值和标准差，最好 epoch 由各 run 的验证表现确定。Pearson、Accuracy 和完整输入 MAE 同时报告，但不得仅挑有利指标。

对于官方 `test`，只在模型、种子集合、超参数、阈值全部冻结后做一次评估。附件 3 不计算指标，也不从 30 条预测反向改模型。

## 5. 指标、规律分析与图表

实现 `src/problem2/evaluate.py`：分类计算 Accuracy、**三类 macro-F1**、每类 precision/recall/F1、混淆矩阵；回归计算 MAE、Pearson。`f1_score` 必须显式指定 `labels=[0,1,2]`、`average='macro'`；Pearson 遇到常数预测时标记为未定义并诊断，不默默写 0。保存逐样本预测以便复核。赛题只写“F1”，论文必须明确本研究使用 macro-F1，并可补充 weighted-F1。

主表至少包含 E0～E3 在完整输入及缺失条件的四项指标（每项三种子均值±标准差）。规律分析至少输出：

1. **缺失模态类型**：分别遮蔽 T/A/V 后指标变化，说明哪一模态缺口最严重；双模态缺失作为补充。
2. **缺失率**：10/30/50/70% 的曲线，横轴是**有效长度占比**，不是 50 个槽位的固定比例。
3. **缺失位置**：开头/中间/结尾相同缺失率的比较。
4. **类别与强度**：负向、中性、正向各自 F1；真实强度接近 0 和绝对值较大的样本分别看 MAE。解释中性易混淆时以混淆矩阵为证。
5. **动态门控行为**：按遮蔽模态与缺失率汇总 `α_T/α_A/α_V`；检查被遮蔽模态平均权重是否合理变化，但**权重变化本身不能证明因果解释**。
6. **错误归因**：选验证集若干代表性错误，列出样本 ID、真值、预测、缺失配置、模态权重、原始转写（仅用于事后分析，不输入模型）。区分模型错误、预处理错误和标签边界问题。

生成 `metric_summary.csv`、`ablation.csv`、`missing_rate.png`、`missing_position.png`、`confusion_matrix.png`、`modality_weights.png`、`error_cases.csv`。若某项曲线不单调，应如实报告并检查样本量、随机视图和置信区间，不强行写“必然下降”。验证视图固定后，可对样本做配对 bootstrap 给主要模型与基线的性能差置信区间。

## 6. 附件 3 最终推理和提交包

选定 E3 或经验证更好的 E4 后，加载**同一**文本编码器、训练集标准化参数和最终学生权重。按 `01` 到 `30` 排序逐文件读取附件 3 对齐版，每文件必须恰好输出一行。推理前检查字段、形状、NaN/Inf；推理后检查概率和约为 1、类别为 0/1/2、强度在 `[-3,3]`，不存在漏文件、重复文件或空预测。建议生成两个文件：

```bash
python -m src.problem2.infer --config configs/problem2.yaml \
  --checkpoint outputs/problem2/final/student.safetensors \
  --input-dir 'data/extracted/E题数据/附件3-模态缺失特征样本/对齐版本' \
  --output outputs/problem2/final/attachment3_predictions_audit.csv
```

- `outputs/problem2/final/attachment3_predictions_audit.csv`：`file_name,class_id,class_name,intensity,p_negative,p_neutral,p_positive,model_id`，用于团队核查。
- `outputs/problem2/final/attachment3_predictions.csv`：按竞赛当时给定的正式列名/模板转换；若题面未给更细的列名规范，至少保留文件标识、极性和强度，并在论文中说明映射。

不对附件 3 编造真实标签或准确率。保存推理命令、权重 SHA256、环境版本和 30 行数量检查结果。若最终附件必须 ≤50 MB，优先提交小型 BERT 的**单份** `safetensors` 权重、仅含自训练层的权重、标准化参数和说明；保存自训练层时不要再把冻结 BERT 完整复制进同一个 checkpoint。模型训练 checkpoint 的优化器状态及教师权重不要放入最终包。打包后执行大小检查，并通过**解压到全新目录、只按提交说明运行一次附件 3 推理**验证可复现性。若无法在 50 MB 内带齐全部模型依赖，准确列明外部通用预训练模型的固定下载地址、revision 与 SHA256，且不得声称包完全离线自包含。

## 7. Agent 执行完成标准

只有以下条件全部满足，才能报告“问题 2 实验完成”：

- 从 `.venv` 可复现安装；PyTorch 设备与 BERT 版本、预训练权重 revision 已记录。
- 数据核查全部通过，训练/验证/附件 2 测试 ID 无交叉；模型输入与附件 3 字段一致。
- E0～E3 均按相同的验证缺失视图完成至少 3 个种子；未完成 E4 时在报告中标为可选未实施。
- 主表、消融、类型/长度/位置分析、错误分析由真实指标文件生成，论文数字能回查。
- 附件 3 对齐版 30 个文件全部产生合法预测，最终模型和数据处理与验证一致。
- 提交包大小符合赛题要求，在干净环境能依说明重跑推理；所有外部模型、软件版本和数据处理规则有明确记录。

## 8. 参考依据

- 赛题本地文档：`复杂场景下多模态情感识别的数学建模与算法设计.docx`。实验边界与提交格式以其正式版本为准。
- [T²DR，ACL 2025](https://aclanthology.org/2025.findings-acl.452/)：模态内部与模态间缺失的分层处理。
- [CMAD，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.html)：缺失模态下的教师学生蒸馏。
- [EBMC，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/He_Enhance-then-Balance_Modality_Collaboration_for_Robust_Multimodal_Sentiment_Analysis_CVPR_2026_paper.html)：按样本调整模态可信度。
- [PyTorch Linux 安装](https://docs.pytorch.org/get-started/locally/)与[Python venv](https://docs.python.org/3.11/tutorial/venv.html)：服务器环境安装依据。
