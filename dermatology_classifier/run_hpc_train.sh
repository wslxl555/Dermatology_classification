#!/bin/bash
# =============================================================================
# 皮肤病「纯疾病三分类」训练 + 评估脚本（dermatology_classifier）
# -----------------------------------------------------------------------------
# 完整流程：
#   Step 0  标签清洗（荨麻疹审核，可选隔离）            —— clean_urticaria_labels.py
#   Step 1  训练三分类模型（预训练 backbone 微调）      —— train.py
#   Step 2  测试集评估（图像级 + 患者级）                —— evaluate.py
#   Step 3  患者级 5-fold 交叉验证（稳健估计 mean±std） —— cross_validate.py
#   Step 4  文件夹批量预测                              —— predict.py
#   Step 5  训练病灶检测器（可解释性，检测头可切换）     —— detector/train_detector.py
#   Step 6  可解释性推理（分类 + 检测 + 病灶特征描述）   —— detector/predict_interpretable.py
#
# 用法：
#   sbatch run_hpc_train.sh     # 提交到调度器（推荐）
#   bash   run_hpc_train.sh     # 在已申请到 GPU 的交互节点 / 容器内直接跑
#
# 注意事项：
#   1) 计算节点通常不联网：先在登录节点缓存预训练权重（见下方「预训练权重缓存」），
#      脚本检测到缓存后会自动给 train / cross_validate 加 --offline。
#   2) 改 BACKBONE 时，同步更新缓存文件名 CACHE_FILE（见下方对照表）。
#   3) 分区 / GPU 资源名因集群而异：若 sbatch 报错，先 sinfo -o "%P %G" 查真实名称
#      再改 --partition 与 --gres。
# =============================================================================

# ---- SLURM 指令（sbatch 提交时生效；bash 交互运行时这些是普通注释）----
#SBATCH --job-name=derm_cls
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
# 训练配置（与 config.py 保持一致）
# =============================================================================
BACKBONE=swin_t            # convnext_tiny / convnext_small / swin_t / resnet50 ...
IMAGE_SIZE=384
EPOCHS=50
CV_EPOCHS=30
FOLDS=5
CLEAN_MOVE=0               # 1=把「需复核」样本隔离到 QUARANTINE_DIR，0=只生成报告
QUARANTINE_DIR=/workspace/quarantine_urticaria
DETECTION_HEAD=fcos        # 病灶检测头：faster_rcnn / retinanet / fcos（可切换）
DET_IMAGE_SIZE=512
DET_EPOCHS=20

set -e  # 任一步出错立即终止，便于尽早暴露问题

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
# 1) 前置检查 1：必须能用到 GPU
# =============================================================================
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("\n[错误] CUDA 不可用 —— 作业落在了 CPU 节点。", file=sys.stderr)
    print("请确认 --partition 与 --gres=gpu:1 已生效，或 sinfo -o '%P %G' 查真实资源名。", file=sys.stderr)
    sys.exit(1)
print(f"GPU 就绪: {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB)")
PY

# =============================================================================
# 2) 前置检查 2：数据目录存在
# =============================================================================
LABELED_DIR=/workspace/labeled
UNLABELED_DIR=/workspace/unlabeled
PREDICT_DIR=/workspace/predict
if [ -d "$LABELED_DIR" ]; then
    echo "训练/验证数据 OK: $(ls "$LABELED_DIR" | tr '\n' ' ')"
else
    echo "[错误] 找不到标注数据目录: $LABELED_DIR"; exit 1
fi
if [ -d "$UNLABELED_DIR" ]; then
    echo "测试数据 OK: $(ls "$UNLABELED_DIR" | tr '\n' ' ')"
else
    echo "[错误] 找不到测试数据目录: $UNLABELED_DIR"; exit 1
fi

# =============================================================================
# 3) 前置检查 3：预训练权重是否已缓存（决定是否加 --offline）
#    缓存路径 = torchvision 默认 ~/.cache/torch/hub/checkpoints/
#    预缓存方法（在有网的登录节点执行一次，对应 BACKBONE 改这行）：
#      python -c "import torchvision.models as m; m.swin_t(weights=m.Swin_T_Weights.IMAGENET1K_V1)"
# =============================================================================
CACHE_DIR=~/.cache/torch/hub/checkpoints
CACHE_FILE="$CACHE_DIR/swin_t-704ceda3.pth"   # ★ 与 BACKBONE 对应的权重文件名
OFFLINE=""
if [ -f "$CACHE_FILE" ]; then
    OFFLINE="--offline"
    echo "预训练权重已缓存，使用 --offline（计算节点不联网）。"
else
    echo "[警告] 未找到缓存权重: $CACHE_FILE"
    echo "       建议先在有网的节点执行上面的预缓存命令。"
    echo "       本次将尝试联网加载；若节点无外网，会退化为随机初始化。"
fi

# 检测器（torchvision detection）用的 ResNet50 权重缓存
DET_CACHE_FILE="$CACHE_DIR/resnet50-0676ba61.pth"
DET_OFFLINE=""
if [ -f "$DET_CACHE_FILE" ]; then
    DET_OFFLINE="--offline"
    echo "检测器 ResNet50 权重已缓存，使用 --offline。"
else
    echo "[警告] 未找到检测器 ResNet50 缓存权重: $DET_CACHE_FILE"
    echo "       建议先缓存: python -c \"from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)\""
fi

# =============================================================================
# Step 0：标签清洗（荨麻疹审核，训练前执行）
#   默认只生成报告 outputs/urticaria_review.csv，供人工复核。
#   把 CLEAN_MOVE 设为 1 时，会把「需复核」样本隔离到 QUARANTINE_DIR，
#   之后训练/评估会自动使用清洗后的数据。
# =============================================================================
echo ""
echo ">>> Step 0: 标签清洗（荨麻疹审核）"
python clean_urticaria_labels.py --disease 荨麻疹 \
    --write-csv outputs/urticaria_review.csv
if [ "$CLEAN_MOVE" = "1" ]; then
    python clean_urticaria_labels.py --disease 荨麻疹 --move "$QUARANTINE_DIR"
fi

# =============================================================================
# Step 1：训练三分类模型
# =============================================================================
echo ""
echo ">>> Step 1: 疾病三分类训练（$BACKBONE @ ${IMAGE_SIZE}px, $EPOCHS epochs）"
python train.py --backbone "$BACKBONE" --image_size "$IMAGE_SIZE" \
    --epochs "$EPOCHS" $OFFLINE

# =============================================================================
# Step 2：测试集评估（图像级 + 患者级）
# =============================================================================
echo ""
echo ">>> Step 2: 测试集评估（图像级 + 患者级）"
python evaluate.py --checkpoint best.pth

# =============================================================================
# Step 3：患者级 k-fold 交叉验证（稳健估计 mean ± std）
# =============================================================================
echo ""
echo ">>> Step 3: 患者级 $FOLDS-fold 交叉验证"
python cross_validate.py --backbone "$BACKBONE" --image_size "$IMAGE_SIZE" \
    --epochs "$CV_EPOCHS" --folds "$FOLDS" $OFFLINE

# =============================================================================
# Step 4：文件夹批量预测
# =============================================================================
echo ""
echo ">>> Step 4: 批量预测 $PREDICT_DIR"
python predict.py --dir "$PREDICT_DIR" --checkpoint best.pth

# =============================================================================
# Step 5：训练病灶检测器（可解释性；检测头可切换）
# =============================================================================
echo ""
echo ">>> Step 5: 训练病灶检测器（$DETECTION_HEAD @ ${DET_IMAGE_SIZE}px, $DET_EPOCHS epochs）"
python detector/train_detector.py --head "$DETECTION_HEAD" \
    --image_size "$DET_IMAGE_SIZE" --epochs "$DET_EPOCHS" $DET_OFFLINE

# =============================================================================
# Step 6：可解释性推理（分类 + 检测 + 病灶特征描述）
# =============================================================================
echo ""
echo ">>> Step 6: 可解释性推理（分类 + 检测 + 病灶特征）"
python detector/predict_interpretable.py --dir "$PREDICT_DIR"

echo ""
echo "============================================"
echo "全部完成 @ $(date)"
echo "============================================"
