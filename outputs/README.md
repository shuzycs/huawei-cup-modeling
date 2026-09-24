# outputs/ 实验产物说明

所有产物由 `src/problem2/` 的脚本生成，**可全部删除后重跑复现**。
本目录不纳入版本控制（见仓库根 `.gitignore`），也不进入提交包
（提交包只取 `final/` 的 CSV、`env/` 的清单与自训练权重）。

## 目录结构

```
outputs/problem2/
├── experiment_summary.md        ★ 实验总结报告（逐 run 明细 + 全部分析）
├── reproduction_report.md       ★ 复现报告（环境、数据核查、消融、提交说明）
├── data_audit.json / data_audit.md   数据核查结果（passed=true，问题数 0）
├── metric_summary.csv           主表：6 个实验 × 三种子聚合（生成：analyze）
├── ablation.csv                 消融表（核心列）
├── analysis_report.json         规律分析的结构化结果（含 36 条件明细）
├── eval_summary_valid.json      验证集评估汇总（生成：evaluate）
├── eval_summary_test.json       test 集评估汇总
│
├── env/            环境清单：environment.json、pip_freeze.txt、
│                   install_commands.txt、data_hashes.json          （生成：report）
├── cache/          训练缓存：train/valid/test.npz、normalizer.npz、
│                   validation_masks.json（129 个固定视图）、
│                   teacher_text.npz（E4 教师用，659 MB）、prepare_manifest.json
│                                                                   （生成：prepare_data）
├── pretrained/     固定 revision 的通用文本编码器（3 个文件 + manifest.json，
│                   共 17.14 MB）                                     （生成：text_encoder）
├── logs/           每个 run 的训练日志 + run_problem2.log           （生成：train）
├── runs/           <E>_seed<S>/：config.json、run_info.json、history.json、
│                   result.json、best.safetensors、last.safetensors、
│                   eval_valid.json、eval_valid_samples.npz、eval_test.json
├── teacher/        E4 教师：mosei_text768_seed<S>/（同上结构）
├── search/         E4 蒸馏权重搜索：E4_<变体>/{eval_valid.json,run_info.json,history.json}
├── tables/         规律分析表（见下）
├── figures/        4 张图（见下）
└── final/          ★ 附件 3 推理结果与自检报告
```

## tables/ 与 figures/

| 文件 | 内容 | 生成 |
|---|---|---|
| `tables/missing_rate.csv` | 缺失率规律，含**实际有效长度占比**区间 | analyze |
| `tables/missing_position.csv` | 位置规律（起/中/末 × 4 个缺失率） | analyze |
| `tables/missing_modality_type.csv` | 模态类型与双模态缺失规律 | analyze |
| `tables/modality_weights.csv` | 动态门控 α_T/α_A/α_V（按视图，三种子均值±标准差） | analyze |
| `tables/class_intensity.csv` | 逐类 F1 与分强度段 MAE | analyze |
| `tables/error_cases.csv` | 60 条代表性错误（含样本 ID、缺失配置、门控权重、原始转写） | analyze |
| `tables/paired_bootstrap.csv` | E5 与各基线、E4 vs E3 的配对 bootstrap 置信区间 | analyze |
| `figures/missing_rate.png` | 缺失率 vs 实际有效长度占比曲线 | analyze |
| `figures/missing_position.png` | 缺失位置对比 | analyze |
| `figures/confusion_matrix.png` | 最终模型完整输入混淆矩阵 | analyze |
| `figures/modality_weights.png` | 动态门控权重 | analyze |

## final/ 提交相关产物

| 文件 | 说明 |
|---|---|
| `attachment3_predictions.csv` | **附件 3 测试集全量预测结果 CSV**（30 行：`file_name,class_id,class_name,intensity`） |
| `attachment3_predictions_audit.csv` | 审计版：增加 `p_negative/p_neutral/p_positive/model_id` |
| `inference_report.json` | 推理记录：权重 SHA256、环境、30 行合法性与自检结果 |

附件 3 **无真实标签**，因此没有任何指标文件；也不存在针对附件 3 的调参记录。

## 数字口径

`x ± s` 中的 `s` 一律是**三种子之间的样本标准差**（ddof=1）。
核对方式见 `tools/README.md`；`python -m tools.audit_report_numbers --strict`
会从本目录的结果文件重算关键数字，并与两份报告交叉比对。

## 体积参考（本机实测）

| 目录 | 大小 | 说明 |
|---|---|---|
| `cache/` | 693 MB | 最大项是 `teacher_text.npz`（659 MB，仅 E4 需要，可删） |
| `runs/` | 606 MB | 大头是各 run 的 `eval_valid_samples.npz` 与 `best.safetensors` |
| `search/` | 163 MB | E4 蒸馏变体搜索（`last.safetensors` 已清理） |
| `pretrained/` | 17 MB | 固定 revision 的通用编码器（3 个文件 + manifest） |
| `teacher/` | 13 MB | 3 个教师权重 |
| 其余 | < 1 MB | 报告、表格、图、日志 |
| **合计** | **约 1.49 GB** | |

## 清理

```bash
python -m tools.cleanup           # 试运行：列出将删除的内容
python -m tools.cleanup --yes     # 执行：删除 __pycache__/.pytest_cache 与未被引用的 last.safetensors
```

`last.safetensors` 是训练中断兜底权重，**代码不读取**（评估与分析只用 `best.safetensors`）；
需要时重跑训练即可再生成。此外已清理 HuggingFace 下载元数据缓存
（`pretrained/*/.cache/`）与早期搜索遗留的临时日志。

## 独立架构实验

outputs/problem2_av_transformer/ 保存音视频时序 Transformer 的 E3/E5 三种子训练与评估结果、比较 JSON 和对应 Markdown。它与默认门控输出分离；结论见 [实验报告](../docs/problem2_av_transformer_comparison.md)。
