"""问题 2 实验：局部模态缺失下的多模态情感预测。

包结构：
    inspect_data  数据核查（只读，产出 data_audit.json/md）
    prepare_data  反序列化 -> .npz 缓存 + 训练集标准化统计
    dataset       Dataset 与批构造
    masks         确定性连续区间缺失生成器 / 固定验证缺失视图
    model         统一模型接口
    train         训练入口
    evaluate      指标计算与逐样本预测
    analyze       规律分析与图表
    infer         附件 3 最终推理
"""

__version__ = "1.0.0"

MODALITIES = ("T", "A", "V")
MODALITY_NAMES = {"T": "text", "A": "audio", "V": "vision"}
NUM_CLASSES = 3
CLASS_NAMES = ("Negative", "Neutral", "Positive")
LABEL_MIN = -3.0
LABEL_MAX = 3.0
