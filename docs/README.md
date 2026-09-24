# docs/ 说明文档

| 文件 | 内容 | 与代码的关系 |
|---|---|---|
| `problem2_experiment_manual.md` | 问题 2 实验手册：赛题边界、数据事实、实验设计、验收标准 | **执行依据**，逐条对应 `src/problem2/` 的实现 |
| `problem2_guide.md` | 运行说明：环境安装、目录结构、运行命令、关键设计约定、实际结果摘要 | 面向"如何跑起来" |
| `problem2_av_transformer_comparison.md` | 音视频两层时序 Transformer 与门控的三种子重训对照 | 冻结和解冻文本编码器下的收益判断 |
| `problem2_cross_attention_comparison.md` | 双层交叉注意力与门控的三种子重训对照、成对区间 | 新架构是否提升的实测结论 |
| `problem_analysis.md` | 赛题整体分析（问题 1/2/3 的任务拆解、数据字段、术语） | 背景资料 |
| `terminology.md` | 赛题术语表 | 背景资料 |
| `problem_statement.docx` | 赛题原文 | 权威依据 |

## 阅读顺序建议

1. **想了解做了什么、结果如何** → `../outputs/problem2/experiment_summary.md`
2. **想复现实验** → `problem2_guide.md` + 仓库根 `README.md`
3. **想了解为什么这样设计** → `problem2_experiment_manual.md`
4. **想核对数字真伪** → `../tools/README.md` 中的 `audit_report_numbers`

## 与手册的差异（重要）

实验手册假设 Linux 服务器与 `python3.11 -m venv`，并假设数据目录含 `E题数据/` 中间层。
本机实际情况与之不同，已在两处修正并在报告中记录：

| 项 | 手册假设 | 本机实际 | 处理 |
|---|---|---|---|
| 平台 | Linux + bash | Windows 10 + PowerShell | 提供 `scripts/run_problem2.ps1` 与 `.sh` 两套脚本 |
| 虚拟环境 | `python3.11 -m venv .venv` | conda 环境 `shuzy-hcm`（Python 3.11.16） | 记录于 `outputs/problem2/env/` |
| 数据路径 | `data/extracted/E题数据/附件2-...` | `data/extracted/附件2-...`（无中间层） | `configs/problem2.yaml` 按实际路径填写 |

手册正文中的路径示例保持原样未改（它是执行依据的历史记录），
实际生效的路径以 `configs/problem2.yaml` 为准。
