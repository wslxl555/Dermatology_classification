#!/bin/bash
# =============================================================================
# 「病灶 crop 分类 + 晚融合」完整流程
# -----------------------------------------------------------------------------
# 在整图 CNN 基础上，把病灶形态特征（来自 LabelMe 标注 / crop 形态分类器）融入
# 疾病判断，做患者级 5-fold 对照，回答"形态是否互补、融合能否超过整图 CNN"。
#
# 完整流程：
#   Step 1  整图 CNN 训练（若已有 best.pth 则跳过）           —— train.py
#   Step 2  病灶形态分类器训练                                —— train_lesion_classifier.py
#   Step 3  晚融合对照（冻结 backbone + GT 形态，无泄漏）      —— late_fusion.py
#   Step 4  晚融合对照（微调 best.pth + GT 形态，上界）        —— late_fusion.py --checkpoint
#   Step 5  两段式疾病预测（crop 分类器预测形态 → 疾病）       —— predict_by_lesions.py
#
# 用法：
#   bash run_fusion.sh     （在已申请到 GPU 的节点 / 容器内直接跑）
#
# 注意：
#   1) 计算节点通常不联网：脚本检测到缓存的 swin_t 权重后自动加 --offline。
#   2) 改 BACKBONE 时同步更新 CACHE_FILE 文件名。
# =============================================================================

# ---- SLURM 指令（sbatch 提交时生效；bash 交互运行时这些是普通注释）----
#SBATCH --job-name=derm_fusion
#SBATCH --partition=GPU1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# =============================================================================
# 训练配置
# =============================================================================
BACKBONE=swin_t
IMAGE_SIZE=384            # 整图 CNN 输入
CROP_IMAGE_SIZE=224       # 病灶 crop 输入
EPOCHS=50                 # 整图 CNN 训练轮数
CROP_EPOCHS=20            # 形态分类器训练轮数
FOLDS=5

set -e

cd /workspace/dermatology_classifier || { echo "[错误] 项目目录不存在"; exit 1; }

# =============================================================================
# 0) 打印环境信息
# =============================================================================
echo "============================================"
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "Python:   $(python --version)"
echo "PyTorch:  $(python -c 'import torch; print(torch.__version__)')"
echo "CUDAtorch:$(python -c 'import torch; print(torch.version.cuda)')"
echo "============================================"

# =============================================================================
# 1) 前置检查：GPU
# =============================================================================
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("\n[错误] CUDA 不可用 —— 作业落在了 CPU 节点。", file=sys.stderr)
    sys.exit(1)
print(f"GPU 就绪: {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB)")
PY

# =============================================================================
# 2) 前置检查：数据目录
# =============================================================================
LABELED_DIR=/workspace/labeled
if [ -d "$LABELED_DIR" ]; then
    echo "训练/验证数据 OK: $(ls "$LABELED_DIR" | tr '\n' ' ')"
else
    echo "[错误] 找不到标注数据目录: $LABELED_DIR"; exit 1
fi
# 标注 JSON 是否存在（融合形态的前提）
N_JSON=$(find "$LABELED_DIR" -name '*.json' | wc -l)
echo "病灶标注 JSON 数: $N_JSON"
if [ "$N_JSON" -eq 0 ]; then
    echo "[错误] labeled/ 下没有 LabelMe 标注，无法做形态融合。"; exit 1
fi

# =============================================================================
# 3) 前置检查：预训练权重缓存（决定是否加 --offline）
# =============================================================================
CACHE_DIR=~/.cache/torch/hub/checkpoints
CACHE_FILE="$CACHE_DIR/swin_t-704ceda3.pth"   # ★ 与 BACKBONE 对应的权重文件名
OFFLINE=""
if [ -f "$CACHE_FILE" ]; then
    OFFLINE="--offline"
    echo "预训练权重已缓存，使用 --offline（计算节点不联网）。"
else
    echo "[警告] 未找到缓存权重: $CACHE_FILE"
    echo "       本次将尝试联网加载；若节点无外网，会退化为随机初始化。"
fi

# =============================================================================
# Step 1：整图 CNN 训练（若已有 best.pth 则跳过）
# =============================================================================
echo ""
if [ -f outputs/checkpoints/best.pth ]; then
    echo ">>> Step 1: 跳过（已存在 outputs/checkpoints/best.pth）"
else
    echo ">>> Step 1: 整图 CNN 训练（$BACKBONE @ ${IMAGE_SIZE}px, $EPOCHS epochs）"
    python train.py --backbone "$BACKBONE" --image_size "$IMAGE_SIZE" \
        --epochs "$EPOCHS" $OFFLINE
fi

# =============================================================================
# Step 2：病灶形态分类器训练
# =============================================================================
echo ""
echo ">>> Step 2: 病灶形态分类器训练（$BACKBONE @ ${CROP_IMAGE_SIZE}px, $CROP_EPOCHS epochs）"
python "crop_classifier/train_lesion_classifier.py" \
    --backbone "$BACKBONE" --image_size "$CROP_IMAGE_SIZE" \
    --epochs "$CROP_EPOCHS" $OFFLINE

# =============================================================================
# Step 3：晚融合对照（冻结 backbone + GT 形态，无泄漏）
# =============================================================================
echo ""
echo ">>> Step 3: 晚融合对照（冻结 $BACKBONE + GT 形态）"
python "crop_classifier/late_fusion.py" \
    --backbone "$BACKBONE" --image_size "$CROP_IMAGE_SIZE" $OFFLINE

# =============================================================================
# Step 4：晚融合对照（微调 best.pth + GT 形态，上界）
# =============================================================================
echo ""
echo ">>> Step 4: 晚融合对照（微调 best.pth + GT 形态）"
python "crop_classifier/late_fusion.py" \
    --checkpoint outputs/checkpoints/best.pth

# =============================================================================
# Step 5：两段式疾病预测（crop 分类器预测形态 → 疾病）
# =============================================================================
echo ""
echo ">>> Step 5: 两段式疾病预测（crop 形态分类器 → 疾病，患者级 CV）"
python "crop_classifier/predict_by_lesions.py" \
    --checkpoint outputs/checkpoints/lesion_morph.pth

echo ""
echo "============================================"
echo "全部完成 @ $(date)"
echo "============================================"
