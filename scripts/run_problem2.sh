#!/usr/bin/env bash
# 问题 2 实验的端到端运行脚本（Linux / bash 版本，与 run_problem2.ps1 等价）。
#
# 用法（仓库根目录）：
#   bash scripts/run_problem2.sh all
#   bash scripts/run_problem2.sh env         # 只写环境清单
#   bash scripts/run_problem2.sh data        # 数据核查 + 准备
#   bash scripts/run_problem2.sh smoke       # 端到端冒烟
#   bash scripts/run_problem2.sh train       # E0..E3 + E5 × seed 42/52/62
#   bash scripts/run_problem2.sh eval        # 固定验证缺失视图评估
#   bash scripts/run_problem2.sh analyze     # 指标表与图
#   bash scripts/run_problem2.sh test        # 附件 2 test 独立检验
#   bash scripts/run_problem2.sh infer       # 附件 3 推理
#
# 每个实验/种子写入独立目录 outputs/problem2/runs/<E>_seed<seed>/，不覆盖上一项。
set -euo pipefail

STAGE="${1:-all}"
CONFIG="${CONFIG:-configs/problem2.yaml}"
PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-42 52 62}"
EXPERIMENTS="${EXPERIMENTS:-E0 E1 E2 E3 E4 E5}"

# E5 的验证集小范围搜索选定覆盖项（记录在各自 run_info.json 的 overrides 字段）
E5_OVERRIDES=(--set text_model.unfrozen_learning_rate=0.001
              --set training.class_weighted_loss=true
              --set text_model.min_lr_ratio=0.2)
# E4 uses the recorded final config: KL weight 0.0, temperature 4.0.
E4_OVERRIDES=(--teacher)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONIOENCODING=utf-8

LOG_DIR="outputs/problem2/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_problem2.log"

run_py() {
  local label="$1"; shift
  echo "[$(date '+%F %T')] $label :: $PYTHON $*" | tee -a "$LOG_FILE"
  local start=$SECONDS
  "$PYTHON" "$@" 2>&1 | tee -a "$LOG_FILE"
  local code=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] $label exit=$code elapsed=$((SECONDS - start))s" | tee -a "$LOG_FILE"
  if [ "$code" -ne 0 ]; then
    echo "阶段 '$label' 失败（exit=$code）" >&2
    exit "$code"
  fi
}

echo "Python: $("$PYTHON" -c 'import sys;print(sys.executable)')"
"$PYTHON" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

case "$STAGE" in
  all|env)
    run_py "环境清单" -m src.problem2.report --config "$CONFIG"
    ;;&
  all|data)
    run_py "数据核查"   -m src.problem2.inspect_data --config "$CONFIG"
    run_py "数据准备"   -m src.problem2.prepare_data --config "$CONFIG"
    ;;&
  all|smoke)
    run_py "冒烟测试" -m src.problem2.train --config "$CONFIG" --experiment E3 --seed 42 --smoke --max-epochs 1
    ;;&
  all|train)
    for exp in $EXPERIMENTS; do
      for seed in $SEEDS; do
        if [ "$exp" = "E4" ]; then
          # 教师必须先在相同 seed 上训练好（已训练则跳过）
          teacher_ckpt="outputs/problem2/teacher/mosei_text768_seed${seed}/best.safetensors"
          if [ ! -f "$teacher_ckpt" ]; then
            run_py "训练教师 seed=$seed" -m src.problem2.train --config "$CONFIG" --experiment E4 --seed "$seed" --train-teacher
          fi
          run_py "训练 $exp seed=$seed" -m src.problem2.train --config "$CONFIG" --experiment "$exp" --seed "$seed" "${E4_OVERRIDES[@]}"
        elif [ "$exp" = "E5" ]; then
          run_py "训练 $exp seed=$seed" -m src.problem2.train --config "$CONFIG" --experiment "$exp" --seed "$seed" "${E5_OVERRIDES[@]}"
        else
          run_py "训练 $exp seed=$seed" -m src.problem2.train --config "$CONFIG" --experiment "$exp" --seed "$seed"
        fi
      done
    done
    ;;&
  all|eval)
    for exp in $EXPERIMENTS; do
      for seed in $SEEDS; do
        run_py "评估 $exp seed=$seed" -m src.problem2.evaluate --config "$CONFIG" --split valid --experiment "$exp" --seed "$seed"
      done
    done
    ;;&
  all|analyze)
    run_py "规律分析与图表" -m src.problem2.analyze --config "$CONFIG"
    ;;&
  all|test)
    for exp in $EXPERIMENTS; do
      run_py "附件2 test 检验 $exp" -m src.problem2.evaluate --config "$CONFIG" --split test --experiment "$exp" --seed 42
    done
    ;;&
  all|infer)
    run_py "附件3 推理" -m src.problem2.infer --config "$CONFIG" --experiment E5 \
      --checkpoint outputs/problem2/runs/E5_seed42/best.safetensors \
      --input-dir "data/extracted/附件3-模态缺失特征样本/对齐版本" \
      --output outputs/problem2/final/attachment3_predictions_audit.csv \
      --submission-output outputs/problem2/final/attachment3_predictions.csv
    ;;&
  *)
    ;;
esac

echo "阶段 '$STAGE' 完成。日志：$LOG_FILE"
