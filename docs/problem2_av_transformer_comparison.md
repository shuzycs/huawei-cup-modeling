# 音视频两层时序 Transformer 对照实验

## 改动与设计

音频 74 维、视觉 35 维各自投影到 256 维，加模态内部位置编码，分别通过两层 4 头时序 Transformer（前馈维度 512）。每层用可观测掩码屏蔽缺失位置；随后逐位置映射回 128 维，沿用原有掩码注意力池化和动态门控。文本编码器、融合方式、训练损失、缺失增强和数据划分均按对应 E3/E5 基线保持一致。实验配置为 [problem2_av_transformer.yaml](../configs/problem2_av_transformer.yaml)。

这种设计让每个可观测音视频位置在池化前读取同模态的前后文，适合长程声学变化和表情过程；代价是模型主体从 173,704 增至 2,264,328 个参数，约 13 倍，在 3,395 条训练样本上有过拟合风险。

## 训练与评估

在种子 42、52、62 上完整重训 E3 与 E5，保留历史超参数和相同固定缺失视图。E5 使用历史最佳设置：文本后两层学习率 0.001、分类加权、最小学习率比例 0.2。验证集有 728 条样本，每条包含完整输入和固定缺失视图；主指标为 36 个单模态缺失条件的 macro-F1。独立 test 只报告完整输入指标。模型按训练时验证指标选择最佳 checkpoint。历史门控权重没有重训或覆盖。

| 模型 | 验证完整输入 F1 | 验证缺失 36 条件 F1 | 验证缺失 MAE | 独立 test F1 | 独立 test MAE |
|---|---:|---:|---:|---:|---:|
| E3 门控 | 0.5330 | 0.5199 | 0.7290 | 0.5174 | 0.7897 |
| E3 时序 Transformer | **0.5528** | **0.5378** | **0.6996** | **0.5373** | **0.7764** |
| E5 门控（当前默认） | **0.5733** | **0.5583** | **0.6990** | **0.5541** | **0.7489** |
| E5 时序 Transformer | 0.5629 | 0.5449 | 0.6993 | 0.5425 | 0.7728 |

数值为三种子均值；F1 越高越好，MAE 越低越好。逐种子标准差、差值和按样本聚类的配对区间见 [完整机器生成报告](../outputs/problem2_av_transformer/comparison_with_gate.md)。

- **E3 有收益**：缺失 36 条件 F1 比同设定门控高 0.0178，独立 test F1 高 0.0198。缺失条件 MAE 低 0.0294。三类单模态缺失条件的 F1 均值均提升；文本缺失 70% 的九个种子×位置组合中有 5 个改善，平均提升 0.0095。
- **E5 无收益**：缺失 36 条件 F1 低 0.0134，独立 test F1 低 0.0116，test MAE 高 0.0239。三类单模态缺失条件的 F1 均值都下降，文本缺失 70% 平均下降 0.0182。
- **总体选择**：E3 时序 Transformer 虽优于 E3 门控，仍低于 E5 门控。因此默认模型和最终附件 3 权重维持 E5 门控。可在冻结文本编码器或需要研究音视频长程上下文时使用该变体。

配对验证集准确率差的 95% 样本聚类区间：E3 缺失条件为 [+0.0084, +0.0411]，E5 为 [-0.0406, -0.0068]。这些区间固定了已选 checkpoint；因为 checkpoint 选取使用同一验证集，区间仅作描述，不能视为独立泛化显著性检验。独立 test 仅有完整输入，不能证明 test 缺失场景的表现。

## 复现

~~~powershell
$py = "D:\WorkSoftware\anaconda\envs\shuzy-hcm\python.exe"
foreach ($seed in 42,52,62) {
    & $py -m src.problem2.train --config configs/problem2_av_transformer.yaml --experiment E3 --seed $seed
    & $py -m src.problem2.train --config configs/problem2_av_transformer.yaml --experiment E5 --seed $seed --set text_model.unfrozen_learning_rate=0.001 --set training.class_weighted_loss=true --set text_model.min_lr_ratio=0.2
}
foreach ($split in "valid","test") {
    foreach ($experiment in "E3","E5") {
        & $py -m src.problem2.evaluate --config configs/problem2_av_transformer.yaml --split $split --experiment $experiment
    }
}
& $py -m tools.compare_av_transformer
~~~

训练日志、checkpoint、逐种子评估和比较 JSON 保存在 outputs/problem2_av_transformer/。比较脚本会核对训练配置及验证样本/视图是否匹配。
