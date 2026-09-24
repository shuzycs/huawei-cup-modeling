# 问题 2 运行说明（局部模态缺失下的情感预测）

**架构对照（2026-09-24）**：两层交叉注意力与可学习融合 token 的三种子重训结果未提升验证缺失主指标。默认 configs/problem2.yaml 继续使用门控；实验配置为 configs/problem2_cross_attention.yaml，新实验产物在 outputs/problem2_cross_attention/。下方历史分数和附件 3 预测均属于默认门控模型；新旧完整对照见该目录的 comparison_with_gate.md。

本目录实现《问题 2：局部模态缺失下的情感预测实验手册》的全部流程：数据核查 → 缓存与标准化 →
训练（E0/E1/E2/E3 × 3 种子）→ 固定缺失视图评估 → 规律分析与图表 → 附件 3 最终推理。

- 本次实际执行环境为 **Windows 10 + PowerShell**（手册假设 Linux/bash）。代码本身与平台无关，
  只依赖 `pathlib` 与相对路径，因此同一套代码在 Linux 上把 `python` 换成 `python3.11 -m venv .venv` 即可。
- 路径全部相对**仓库根目录**，脚本内部不出现任何绝对路径或 `D:\` 盘符。
- 本机数据目录实际为 `data/extracted/附件2-数据集特征文件/`、`data/extracted/附件3-模态缺失特征样本/对齐版本/`
  （**没有**手册示例里的 `E题数据/` 中间层），因此 `configs/problem2.yaml` 已按实际路径填写。

## 1. 环境

```powershell
# 1) 创建/重建 conda 环境（若同名环境已存在，先删除再创建）
conda env remove -n shuzy-hcm -y
conda create -n shuzy-hcm python=3.11 -y

# 2) 安装 CUDA 版 PyTorch（RTX 4090，驱动 560.94 / CUDA 12.6，选择 cu118 轮子）
$py = "$env:USERPROFILE\anaconda3\envs\shuzy-hcm\python.exe"   # 本机实际为 D:\WorkSoftware\anaconda\envs\shuzy-hcm\python.exe
& $py -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.4.0

# 3) 其余依赖（务必在 PyTorch 之后执行；numpy 必须 <2，torch 2.4.0 由 NumPy 1.x 编译）
& $py -m pip install "numpy>=1.26,<2" "pandas>=2.2,<3" "openpyxl>=3.1,<4" "scikit-learn>=1.4,<2" `
  "transformers>=4.40,<5" "huggingface_hub>=0.26,<2" "safetensors>=0.4,<1" "matplotlib>=3.8,<4" `
  "PyYAML>=6,<7" "tqdm>=4.66,<5" "pytest>=8,<10"

# 4) 记录环境清单
& $py -m src.problem2.report --config configs/problem2.yaml
```

实际执行的命令、版本与设备信息见 `outputs/problem2/env/`：
`install_commands.txt`、`pip_freeze.txt`、`environment.json`、`data_hashes.json`。

本次核实结果：Python 3.11.16 / torch 2.4.0+cu118 / CUDA 11.8 build / cuDNN 9.1.0 /
`torch.cuda.is_available() == True` / `NVIDIA GeForce RTX 4090`（23.99 GiB, sm_89）。

> 注意：`numpy>=2` 会让 torch 2.4.0 报 `Numpy is not available`，因此 `pip_freeze.txt` 中 numpy 为 1.26.x。

## 2. 预训练文本编码器（已固定 revision）

| 项目 | 值 |
|---|---|
| repo_id | `google/bert_uncased_L-2_H-128_A-2` |
| revision（commit SHA） | `30b0a37ccaaa32f332884b96992754e246e48c5f` |
| model.safetensors | 17,739,144 B，SHA256 `7fb69ad9f6866d8983183c930e33828f326470bf6ad8bbb2ad4ed957a92e9414` |
| vocab.txt | 231,508 B，SHA256 `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` |
| config.json | 382 B，SHA256 `508e1f01aae55d73355cbdd82609be2f43ba5a0d3428837adbe56cf8391f8b39` |

该模型是**通用小型 BERT，不是情感微调模型**；本地目录只保留推理必需的 3 个文件（无 `.bin` 冗余），
总 17.14 MB。下载命令：

```powershell
& $py -m src.problem2.text_encoder --config configs/problem2.yaml --write-config-revision
```

## 3. 目录结构

```
src/problem2/
  common.py        # 配置、稳定随机种子、JSON/日志、环境信息
  data.py          # 附件2/附件3 读取、label.xlsx 按 ID 连接
  inspect_data.py  # 数据核查 -> outputs/problem2/data_audit.{json,md}
  prepare_data.py  # 划分缓存 .npz、训练集标准化统计、固定验证缺失视图
  masks.py         # SpanMasker（连续区间缺失）、固定验证视图、缺失应用
  dataset.py       # SplitData、批构造、TextEmbedder（冻结/解冻 BERT）
  text_encoder.py  # BERT 下载/revision 固定/SHA256 清单
  model.py         # 统一模型（历史门控或双层交叉注意力 + 融合 token + 双任务）
  train.py         # 训练（E0..E5，三种子，支持 --set 覆盖）
  metrics.py       # macro-F1 / 每类指标 / MAE / Pearson / 配对 bootstrap
  evaluate.py      # 固定缺失视图评估、逐样本预测输出、附件2 test 独立检验
  analyze.py       # 主表/消融表/规律分析/图表/错误分析
  infer.py         # 附件 3 最终推理与自检
  report.py        # 环境清单
tests/test_problem2_core.py   # 掩码/文本遮蔽/池化/融合权重/前后向 单元检查（20 项）
scripts/run_problem2.ps1      # Windows 端到端运行脚本
scripts/run_problem2.sh       # Linux 端到端运行脚本
configs/problem2.yaml         # 配置（revision 已固定）
outputs/problem2/reproduction_report.md   # 复现报告（全部真实指标）
```

## 4. 运行

```powershell
# 全部阶段（数据核查 → 准备 → 冒烟 → E0..E3/E5 × seed 42/52/62 → 评估 → 分析 → test 检验 → 附件3 推理）
powershell -ExecutionPolicy Bypass -File scripts/run_problem2.ps1 -Stage all

# Linux 等价脚本
bash scripts/run_problem2.sh all

# 或分阶段
powershell -ExecutionPolicy Bypass -File scripts/run_problem2.ps1 -Stage data
powershell -ExecutionPolicy Bypass -File scripts/run_problem2.ps1 -Stage smoke
powershell -ExecutionPolicy Bypass -File scripts/run_problem2.ps1 -Stage train -Experiments E0,E1,E2,E3,E5 -Seeds 42,52,62
powershell -ExecutionPolicy Bypass -File scripts/run_problem2.ps1 -Stage eval
powershell -ExecutionPolicy Bypass -File scripts/run_problem2.ps1 -Stage analyze
powershell -ExecutionPolicy Bypass -File scripts/run_problem2.ps1 -Stage test
powershell -ExecutionPolicy Bypass -File scripts/run_problem2.ps1 -Stage infer
```

等价的单条命令（与手册一致）：

```powershell
& $py -m src.problem2.inspect_data --config configs/problem2.yaml
& $py -m src.problem2.prepare_data --config configs/problem2.yaml
& $py -m src.problem2.train --config configs/problem2.yaml --experiment E3 --seed 42
& $py -m src.problem2.evaluate --config configs/problem2.yaml --split valid --all-experiments
& $py -m src.problem2.analyze --config configs/problem2.yaml
& $py -m src.problem2.infer --config configs/problem2.yaml `
    --checkpoint outputs/problem2/runs/E5_seed42/best.safetensors `
    --input-dir "data/extracted/附件3-模态缺失特征样本/对齐版本" `
    --output outputs/problem2/final/attachment3_predictions_audit.csv
& $py -m pytest tests -q
```

小范围超参数搜索用 `--set`（会记录到 run 目录）：

```powershell
& $py -m src.problem2.train --config configs/problem2.yaml --experiment E5 --seed 42 `
    --set text_model.unfrozen_learning_rate=0.001 --set training.class_weighted_loss=true
```

## 5. 输出

```
outputs/problem2/
  env/              环境清单（pip freeze、environment.json、install_commands.txt、data_hashes.json）
  data_audit.json / data_audit.md      数据核查报告（含标签按 ID 连接核对）
  cache/            train/valid/test.npz、normalizer.npz、validation_masks.json（固定验证缺失视图）、
                    teacher_text.npz（E4 教师用的附件 2 预计算 text，659 MB，不提交）
  pretrained/bert_uncased_L-2_H-128_A-2/  固定 revision 的 3 个模型文件 + manifest.json
  teacher/mosei_text768_seed<seed>/       E4 教师（run_info.json、history.json、best/last.safetensors）
  search/E4_<变体>/                        E4 蒸馏权重搜索的每个变体指标（互不覆盖）
  logs/             每个 run 的训练日志 + run_problem2.log
  runs/<E>_seed<seed>/  config.json、run_info.json、history.json、结果、best/last.safetensors、
                        eval_valid.json、eval_valid_samples.npz、eval_test.json
  metric_summary.csv / ablation.csv
  tables/           missing_rate.csv、missing_position.csv、missing_modality_type.csv、
                    modality_weights.csv、class_intensity.csv、error_cases.csv、paired_bootstrap.csv
  figures/          missing_rate.png、missing_position.png、confusion_matrix.png、modality_weights.png
  final/            attachment3_predictions_audit.csv、attachment3_predictions.csv、inference_report.json
  reproduction_report.md                 复现报告（全部真实指标）
```

## 6. 关键设计约定（与手册一致）

1. **输入只允许 `text_bert`**：附件 3 没有预计算 `text`，因此训练也只使用 `text_bert`；附件 2 的
   `text` 仅保留给可选的 E4 教师。
2. **文本缺失在进入 BERT 之前应用**：内容 token 置 `[PAD]`、attention mask 置 0、token type 置 0；
   特殊 token `[CLS]/[SEP]` 不遮蔽，且不计入"有内容"。
3. **有效性 vs 可观测性分开记录**：`v` 来自原始值（文本 attention mask 且非特殊 token；语音/视觉非零行），
   `o` 由缺失生成器给出；局部缺失与原始填充分开统计。
4. **标准化只用训练集可观测行**：`normalizer.npz` 保存每维均值/标准差（std 下界 1e-3），
   验证/附件2 test/附件3 全部使用训练集统计量，标准化之后再次置零。
5. **训练缺失生成**：每批约 25% 样本保留完整输入；其余 70% 遮蔽 1 个模态、30% 遮蔽 2 个模态；
   每个被选模态的区间长度比例从 {0.10,0.30,0.50,0.70} 均匀采样，起点覆盖开头/中部/结尾，
   短样本按比例缩短并保证至少一个内容位置可观测。种子由 `(seed, epoch, sample_index)` 经 SHA256 导出。
6. **固定验证缺失视图**：模态 T/A/V × 位置 起/中/末 × 比例 10/30/50/70% × 3 个随机视图 + 完整视图，
   另有双模态补充组，共 129 个视图；全部由 `(seed, view_id, 样本序号)` 确定性生成并保存在
   `validation_masks.json`，训练中不重新抽样。
7. **模型选择规则（运行前固定）**：以 36 个单模态缺失条件的平均 macro-F1 最大为主准则，
   MAE 较低者优先，并要求完整输入 macro-F1 相比 E1 下降不超过 2 个百分点。
8. **附件 3 不计算指标**，也不根据其预测结果回头调参。

## 6.1 本次实际结果（详见 `outputs/problem2/reproduction_report.md`）

三种子均值：

| 实验 | 完整输入 macro-F1 | 36 缺失条件 macro-F1 |
|---|---|---|
| E0 文本单模态 | 0.5394 | 0.5276 |
| E1 平均池化+拼接MLP | 0.5301 | 0.5166 |
| E2 E1+缺失增强 | 0.5247 | 0.5128 |
| E3 掩码池化+门控 | 0.5330 | 0.5199 |
| E4 E3+教师蒸馏 | 0.5334 | 0.5182 |
| **E5 E3+解冻后两层（最终模型）** | **0.5733** | **0.5583** |

- 冻结文本编码器时，E1→E2（缺失增强）**没有提升**，E2→E3 有小幅提升但不显著；
  纯文本 E0 强于三模态 E1/E2/E3/E4，说明该设定下瓶颈在文本编码器。
- **E4（教师蒸馏）没有通过纳入判据**：36 缺失条件 0.5199 → 0.5182（略降），完整输入基本持平
  （0.5330 → 0.5334），配对 bootstrap 差异 −0.0017、95% CI [−0.0046, +0.0011]（不显著）；
  加入 KL 项的 4 个变体全部劣于不加蒸馏的 E3，只有"去掉 KL、仅保留强度一致性"略好。故不采用 E4。
- E5（解冻 BERT 后两层 + 类别加权 + 文本学习率 1e-3）是唯一显著更好的配置，
  配对 bootstrap 相对 E3 的准确率差 +0.0250（95% CI [0.0218, 0.0280]）。
- 附件 3 的 30 条预测见 `outputs/problem2/final/attachment3_predictions{,_audit}.csv`；
  附件 3 无标签，不计算任何指标。

### 6.2 E4 教师蒸馏的运行方式

```powershell
# 1) 先训练教师（完整输入 + 附件 2 预计算 text (N,50,768)，不使用缺失增强）
& $py -m src.problem2.train --config configs/problem2.yaml --experiment E4 --seed 42 --train-teacher
#    -> outputs/problem2/teacher/mosei_text768_seed42/best.safetensors（valid macro-F1 0.577 ~ 0.603）

# 2) 再用教师蒸馏学生（学生在连续缺失视图上训练）
& $py -m src.problem2.train --config configs/problem2.yaml --experiment E4 --seed 42 --teacher
#    教师路径默认取 outputs/problem2/teacher/mosei_text768_seed<seed>/best.safetensors，
#    也可用 --teacher <path> 显式指定；--set 可覆盖 distillation.* 里的温度与权重。
```

蒸馏损失：`L = CE(真实标签) + 0.5·Huber + kl_weight·KL(教师软分布‖学生)/尺度 + 0.5·强度一致性`。
最终 E4 取 `kl_weight=0.0`（即只用强度一致性），因为带 KL 的所有变体都更差；
KL 默认按 `KL(p_T‖U)` 归一化尺度，以免不同教师的置信度差异让权重不可比。

## 7. 复现与提交

- `outputs/problem2/final/` 中的审计 CSV 与 `inference_report.json` 记录了推理命令、权重 SHA256、
  环境版本与 30 行数量检查结果。
- 提交包只包含 `src/`、`configs/`、`scripts/`、`tests/`、本说明、轻量权重与 CSV；
  **不打包** `data/`、虚拟环境、缓存、优化器状态和大日志（见 `.gitignore`）。
- 冻结 BERT 权重单独提供 `model.safetensors`（17.7 MB），自训练层为各自的 `best.safetensors`（< 2 MB），
  两者不重复打包。若需完全离线，请按第 2 节的 revision 与 SHA256 自行准备该模型。
