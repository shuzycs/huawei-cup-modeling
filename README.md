# 复杂场景下多模态情感识别的数学建模与算法设计

华为杯数学建模竞赛 E 题工作仓库。**本仓库当前只包含问题 2 的完整实现与实验。**

问题 2 要求：在文本、语音、视觉模态出现**局部连续缺失**的条件下，稳定输出
**三分类情感极性**与 **[-3, 3] 连续情感强度**，分析缺失类型/位置/时长的影响规律，
并对附件 3 的 30 个无标签样本给出最终预测。

## 目录结构

```
.
├── README.md                    # 本文件：总览与快速开始
├── requirements.txt             # 直接依赖的版本区间（含 PyTorch 安装说明）
├── requirements.lock.txt        # 本机实际安装的直接依赖精确版本
├── .gitignore / .gitattributes  # 版本控制规则
│
├── src/problem2/                # 源码包（只用相对路径，跨平台）
│   ├── common.py                # 配置加载、稳定随机种子、JSON/日志、环境信息
│   ├── data.py                  # 附件 2/附件 3 读取、label.xlsx 按 ID 连接
│   ├── masks.py                 # SpanMasker（连续区间缺失）、固定验证视图、缺失应用
│   ├── dataset.py               # 缓存读取、批构造、TextEmbedder（冻结/解冻 BERT）
│   ├── text_encoder.py          # 预训练编码器下载 / revision 固定 / SHA256 清单
│   ├── model.py                 # 统一模型：掩码时序编码 + 门控或双层交叉注意力 + 双任务输出
│   ├── inspect_data.py          # ① 数据核查
│   ├── prepare_data.py          # ② 缓存与标准化
│   ├── train.py                 # ③ 训练 E0~E5（含 E4 教师与学生）
│   ├── evaluate.py              # ④ 固定缺失视图评估、逐样本预测、附件 2 test 检验
│   ├── metrics.py               # macro-F1 / 每类指标 / MAE / Pearson / 配对 bootstrap
│   ├── analyze.py               # ⑤ 主表、消融、规律分析、图表、错误分析
│   ├── infer.py                 # ⑥ 附件 3 推理与自检
│   └── report.py                # 环境清单
│
├── configs/problem2.yaml        # 门控默认配置；另有交叉注意力与音视频时序 Transformer 实验配置
├── scripts/                     # 一键运行脚本（PowerShell + bash）
├── tests/test_problem2_core.py  # 模型与掩码单元检查
├── tools/                       # 只读的诊断/核对工具（见 tools/README.md）
├── docs/                        # 说明文档与赛题原文（见 docs/README.md）
│
├── data/                        # 赛题原始数据（不入库、不入提交包）
└── outputs/                     # 门控、交叉注意力、音视频时序 Transformer 实验各自存放
```

## 快速开始

```powershell
# 1) 创建环境（Windows 用 conda；Linux 可用 python3.11 -m venv .venv）
conda env remove -n shuzy-hcm -y
conda create -n shuzy-hcm python=3.11 -y
$py = "$env:USERPROFILE\anaconda3\envs\shuzy-hcm\python.exe"

# 2) 安装依赖（PyTorch 必须先装，且 numpy 必须 <2）
& $py -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.4.0
& $py -m pip install -r requirements.txt

# 3) 分阶段运行（每步都可单独重跑；产物写入独立目录，不覆盖上一项）
& $py -m src.problem2.report         --config configs/problem2.yaml   # 环境清单
& $py -m src.problem2.inspect_data   --config configs/problem2.yaml   # 数据核查
& $py -m src.problem2.prepare_data   --config configs/problem2.yaml   # 缓存 + 固定验证视图
& $py -m src.problem2.train --config configs/problem2.yaml --experiment E5 --seed 42
& $py -m src.problem2.evaluate --config configs/problem2.yaml --split valid --all-experiments
& $py -m src.problem2.analyze        --config configs/problem2.yaml
& $py -m src.problem2.infer --config configs/problem2.yaml `
    --checkpoint outputs/problem2/runs/E5_seed42/best.safetensors `
    --output outputs/problem2/final/attachment3_predictions_audit.csv

# 或一键跑完整流程（含 E4 教师蒸馏）
powershell -ExecutionPolicy Bypass -File scripts/run_problem2.ps1 -Stage all
bash scripts/run_problem2.sh all          # Linux 等价

# 4) 自检（编译 → 单元测试 → 报告数字核对 → 工具冒烟）
& $py -m tools.check_all
& $py -m tools.collect_evidence          # 打印全部真实证据
& $py -m tools.cleanup                   # 清理缓存（默认试运行）
```

> **注意**：本机 GPU 为 RTX 4090（驱动 560.94 / CUDA 12.6），选用 cu118 轮子。
> 若换机器，请按 [PyTorch 官方选择器](https://docs.pytorch.org/get-started/locally/)重新选择，
> **不要照抄 CUDA 版本**。

## 关键结果（三种子均值，全部来自真实日志）

下表是默认门控配置的三种子结果。双层交叉注意力已用相同种子、数据和验证缺失视图重训，其配置单独保存在 configs/problem2_cross_attention.yaml，详细对照见下文。当前默认配置和最终附件 3 权重仍采用表现更好的门控结构。

| 实验 | 说明 | 完整输入 macro-F1 | 36 缺失条件 macro-F1 |
|---|---|---|---|
| E0 | 文本单模态 | 0.5394 ± 0.0023 | 0.5276 ± 0.0016 |
| E1 | 固定 50 位平均 + 拼接 MLP | 0.5301 ± 0.0121 | 0.5166 ± 0.0109 |
| E2 | E1 + 连续区间缺失训练 | 0.5247 ± 0.0092 | 0.5128 ± 0.0070 |
| E3 | E2 + 掩码池化 + 动态门控 | 0.5330 ± 0.0108 | 0.5199 ± 0.0118 |
| E4 | E3 + 完整输入教师蒸馏 | 0.5334 ± 0.0176 | 0.5182 ± 0.0143 |
| **E5** | **E3 + 解冻文本编码器后两层（最终模型）** | **0.5733 ± 0.0044** | **0.5583 ± 0.0005** |

三个必须如实说明的结论：

1. **连续区间缺失增强单独使用无效**：E1 → E2 主准则从 0.5166 降到 0.5128。
2. **纯文本 E0 强于所有冻结编码器的三模态模型**（E1/E2/E3/E4）——瓶颈在 4.43M 参数的通用小型 BERT；
   换更强的文本编码器应能显著提升，但与"提交包 ≤ 50 MB"冲突。
3. **E4 教师蒸馏未通过纳入判据**：E4 vs E3 的配对 bootstrap 准确率差 −0.0017，
   95% CI [−0.0046, +0.0011]，不显著；带 KL 项的 4 个蒸馏变体全部劣于不蒸馏的 E3。

最终模型 E5 相对 E3 的准确率差 +0.0250（95% CI [+0.0218, +0.0280]，显著）。
附件 3 的 30 条预测已生成并通过自检（**无真实标签，不计算任何指标**）。
提交包 19.74 MB，干净目录复现验证逐行完全一致。

## 双层交叉注意力重训结论

E3、E4、E5 均按旧实验的三个种子（42/52/62）、相同训练超参数和固定缺失图谱重训。
下表为 macro-F1 的三种子均值；箭头左侧是门控，右侧是交叉注意力。

| 实验 | 验证集 36 缺失条件 | 差值 | 独立 test 完整输入 | 差值 |
|---|---:|---:|---:|---:|
| E3 | 0.5199 → 0.5078 | −0.0121 | 0.5174 → 0.5033 | −0.0142 |
| E4 | 0.5182 → 0.5097 | −0.0085 | 0.5234 → 0.5281 | +0.0047 |
| E5 | 0.5583 → 0.5256 | −0.0328 | 0.5541 → 0.5485 | −0.0057 |

**结论：新结构未提升主指标。** E5 在三个种子的验证缺失 macro-F1 均下降；
文本缺失 70% 的九个种子×位置组合中有八个下降。E5 的独立 test 强度 MAE
也从 0.7489 升到 0.7861。E4 虽有很小的 test 分类增益，但验证主指标和 test MAE 变差。
因此保留门控模型作为默认和最终模型。逐种子指标、逐样本配对区间及核对方法见
[完整对照报告](docs/problem2_cross_attention_comparison.md)，可用
[比较脚本](tools/compare_fusion_architectures.py)复算。交叉注意力模型的权重与评估结果
保存在 outputs/problem2_cross_attention/，不会覆盖门控实验。

## 音视频时序 Transformer 重训结论

在门控融合保持不变的条件下，音频 74 维和视觉 35 维分别投影至 256 维，各经两层时序 Transformer，再逐位置映射到原有 128 维进行池化。E3/E5 按历史三个种子和固定验证缺失视图完整重训。

| 实验 | 验证集 36 缺失条件 F1（门控 → 时序 Transformer） | 独立 test 完整输入 F1（门控 → 时序 Transformer） |
|---|---:|---:|
| E3 | 0.5199 → **0.5378**（+0.0178） | 0.5174 → **0.5373**（+0.0198） |
| E5 | **0.5583** → 0.5449（−0.0134） | **0.5541** → 0.5425（−0.0116） |

E3 明显受益，但 E5 退步，且 E3 新结构仍未超过 E5 门控。因此默认模型和最终权重维持 E5 门控。参数量从约 17 万升至 226 万；完整指标、缺失模态细分和配对区间见[对照报告](docs/problem2_av_transformer_comparison.md)。实验配置为 [problem2_av_transformer.yaml](configs/problem2_av_transformer.yaml)，可用[比较脚本](tools/compare_av_transformer.py)复算。

## 文档索引

| 文档 | 内容 |
|---|---|
| `docs/problem2_experiment_manual.md` | 实验手册（本次执行的依据） |
| `docs/problem2_guide.md` | 运行说明：环境、目录、命令、关键设计约定 |
| `outputs/problem2/experiment_summary.md` | **实验总结报告**（逐 run 明细 + 全部分析） |
| `outputs/problem2/reproduction_report.md` | 复现报告（环境、数据核查、消融、提交说明） |
| `docs/problem_analysis.md` / `docs/terminology.md` | 赛题分析与术语表 |
| `docs/problem_statement.docx` | 赛题原文 |
## 数据说明

`data/` 下为赛题原始附件，**不纳入版本控制、不进入提交包**：

- `data/extracted/附件2-数据集特征文件/aligned_50.pkl`（947.8 MB）、`label.xlsx`
- `data/extracted/附件3-模态缺失特征样本/对齐版本/附件3_01..30.pkl`
- 本仓库只使用**对齐版**；`unaligned_50.pkl` 未参与任何实验。

## 复现性与提交

- 提交包只含代码、配置、说明、轻量权重与 CSV；
  **不打包**原始数据、虚拟环境、缓存、优化器状态与大日志。
- 冻结的通用文本编码器：`google/bert_uncased_L-2_H-128_A-2`，
  revision `30b0a37ccaaa32f332884b96992754e246e48c5f`，
  本地仅保留 `config.json` / `model.safetensors` / `vocab.txt`（17.14 MB）。
- 干净目录复现：只放提交包内容跑一次附件 3 推理，输出与仓库内结果逐行完全一致
  （强度/概率最大绝对差 0.0）。
