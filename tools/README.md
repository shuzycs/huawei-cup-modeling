# tools/ 只读工具

这些脚本**只读取** `outputs/problem2/` 下的产物，用于核对报告数字、检查附件 3 推理、
查看指标与聚合口径。它们不训练模型、不写实验产物（仅在显式指定 `--json` / `--csv` 时导出）。

全部以模块方式运行（`python -m tools.xxx`），也可以直接 `python tools/xxx.py`。

## 工具清单

| 脚本 | 作用 | 典型用法 |
|---|---|---|
| `check_all.py` | **一键自检**：编译 → 单元测试 → 报告数字核对 → 指标表 → 聚合口径 → 附件 3 复核 | `python -m tools.check_all` |
| `collect_evidence.py` | 汇总**全部**实验证据（逐 run 指标、训练信息、教师、蒸馏搜索、test、规律表、门控、附件 3、数据核查、环境） | `python -m tools.collect_evidence`　`--section A B`　`--json /tmp/e.json` |
| `audit_report_numbers.py` | 从结果文件重算关键数字，检查其是否能在报告中原样找到（防止数字漂移/笔误） | `python -m tools.audit_report_numbers --strict` |
| `verify_aggregation.py` | 核对门控权重与缺失位置离散度的聚合口径（先按 run 求均值、再对 run 求 std） | `python -m tools.verify_aggregation` |
| `summary_table.py` | 打印主指标表 + 配对 bootstrap，可另存 CSV | `python -m tools.summary_table --csv /tmp/m.csv` |
| `show_metrics.py` | 打印 test / valid 完整视图指标、混淆矩阵、逐类 F1 | `python -m tools.show_metrics --experiment E5 --seed 42` |
| `show_activations.py` | 统计模型实际使用的激活函数（模块树 + 一次前向的调用来源，含冻结 BERT 内部） | `python -m tools.show_activations --experiment E5`　`--teacher` |
| `show_depth.py` | 统计模型层数（逐子层 / 残差块两种口径）、参数量、冻结与可训练划分 | `python -m tools.show_depth --experiment E5`　`--verbose`　`--teacher` |
| `check_attachment3.py` | 核查附件 3 的可观测性与预测自洽性（各模态有效行数、概率、强度、门控权重） | `python -m tools.check_attachment3` |
| `cleanup.py` | 清理 `__pycache__`、`.pytest_cache` 与未被引用的 `last.safetensors` | `python -m tools.cleanup`（试运行）→ `--yes` 执行 |

## 推荐的日常流程

```bash
python -m tools.cleanup              # 先看会删什么（默认试运行）
python -m tools.check_all            # 改动后一键确认没破坏既有结论
python -m tools.collect_evidence     # 需要看全部原始数字时
```

## 数字核对的使用方式

`audit_report_numbers.py` 的默认行为针对两份报告的不同定位做了区分：

- `outputs/problem2/experiment_summary.md`：**逐项覆盖**全部明细（含逐 run 指标与门控权重）；
- `outputs/problem2/reproduction_report.md`：只核对**汇总级**数字（该报告不逐 run 列指标）。

```bash
# 默认：按各自范围核对，覆盖比例低于 90% 时 --strict 返回非零
python -m tools.audit_report_numbers --strict

# 只看汇总报告、且要求逐 run 明细与门控全部匹配
python -m tools.audit_report_numbers --detail --include-gates --strict

# 核对指定的报告文件
python -m tools.audit_report_numbers --report outputs/problem2/experiment_summary.md --strict
```

## 数值口径约定

为避免"同一个指标两个数"的歧义，本仓库统一约定：

| 记号 | 含义 |
|---|---|
| `x ± s` | `s` 是**三种子（seed 42/52/62）之间的样本标准差**（ddof=1），不是跨条件离散度 |
| 36 缺失条件 macro-F1 | 12 个条件（模态 T/A/V × 位置 起/中/末）的 4 个缺失率各自求条件均值，再对 36 个条件取平均 |
| 条件均值 | 该条件下 3 个固定随机视图的指标平均 |
| 实际有效长度占比 | 来自 `validation_masks.json` 的 `mean_actual_rate`，不是 50 槽位的固定比例 |

`metric_summary.csv` 额外保留 `missing36_macro_f1_condition_std` 列，记录跨 36 个条件的离散度，
与上面的种子标准差区分开。

## 依赖

这些工具共享 `src/problem2/` 的公共模块（配置加载、阈值路径解析等），
因此需要在**仓库根目录**运行，并已激活实验环境（见 `../requirements.txt`）。
