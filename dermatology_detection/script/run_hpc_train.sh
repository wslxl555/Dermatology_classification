#!/bin/bash
# =============================================================================
# 皮肤病多任务（检测 + 疾病分类）HPC 训练脚本
# -----------------------------------------------------------------------------
# 取代旧的 run_training_hpc.sh（旧脚本有 3 个问题，见下方"用法说明"）。
#
# 本脚本完整流程：
#   Step 1  多任务训练：检测（box + annotation）+ 疾病分类   —— train_labeled.py
#   Step 2  皮损统计 → 疾病 分类器训练（秒级）               —— train_annotation_classifier.py
#   Step 3  测试集评估（unlabeled/）+ 端到端推理示例         —— evaluate_unlabeled.py / auto_annotate.py
#
# 用法说明（提交前必读）：
#   0) 两种运行方式：
#        sbatch run_hpc_train.sh     （提交到调度器，推荐）
#        bash   run_hpc_train.sh     （在已申请到 GPU 的交互节点内直接跑）
#
#   1) ★ 上次失败的根因，本脚本已修复 ★
#        (a) 旧脚本跑了 train.py（去扫不存在的 images/sz|tpc|xmz）→ 本脚本用 train_labeled.py
#            （正确入口，直接读 ../labeled 下的图片 + .json 标注）。
#        (b) 作业落在 CPU 节点（CUDA available: False）→ 本脚本指定 GPU 分区 + --gres=gpu:1，
#            并在启动时硬性断言 CUDA 可用，失败立即报错。
#        (c) 分区名写错（--partition=T1 不是 GPU 分区，且导致 "Invalid gres specification"）
#            → 本脚本按《shell 使用手册》用 GPU1 分区。
#
#   2) 分区选择（集群 GPU 队列，见手册 6.1 节）：
#        GPU1 : 4×RTX3090(24G)/节点 —— 本项目够用，默认选它
#        GPU2 : 8×RTX3090(24G)/节点
#        GPU3 : 8×A800(80G)/节点    —— 显存要求高时才用
#      若 GPU1 队列排队长，可把下面 --partition=GPU1 改成 GPU2。
#
#   3) 提交前在【登录节点】先缓存预训练权重（登录节点有网，计算节点通常没网）：
#        cd <项目目录>
#        python download_weights.py --backbone convnext_tiny
#
#   4) 确认下面 PROJECT_DIR / CONDA_ENV 两个变量与你的实际路径一致。
#
#   5) 监控：squeue -u $USER；日志：tail -f slurm_<jobid>.out / outputs/*.log
#
#   6) 若 --gres=gpu:1 仍报错，先确认本集群真实的分区/GPU 资源名（不同集群可能不同）：
#        sinfo -o "%P %G %a %N"     # 查看每个分区的 GRES 名称与可用性
#        scontrol show partition     # 查看分区详情
#      然后把 --partition 和 --gres 改成上面命令显示的真实名称。
# =============================================================================

# ---- SLURM 指令（sbatch 提交时生效；bash 交互运行时这些是普通注释）----
#SBATCH --job-name=derm_train
#SBATCH --partition=T1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# =============================================================================
# 用户需确认的变量
# =============================================================================
#PROJECT_DIR=~/Dermatology_classification/dermatology_detection   # ★ 改成你的实际路径
#CONDA_ENV=dermatology_classifier                                 # conda 环境名
BACKBONE=convnext_tiny
DETECTION_HEAD=fcos                                      # fcos / faster_rcnn / cascade_rcnn / yolov8 / rtdetr
IMAGE_SIZE=512
EPOCHS=50
BATCH_SIZE=8

set -e  # 任一步出错立即终止，便于尽早暴露问题

# =============================================================================
# 0) 初始化 conda 环境
# =============================================================================
#source /phadcloud/home/software/miniforge/etc/profile.d/conda.sh
#conda activate "$CONDA_ENV"

cd /workspace/dermatology_detection || { echo "[错误] 项目目录不存在: $PROJECT_DIR"; exit 1; }

# =============================================================================
# 1) 打印环境信息
# =============================================================================
echo "============================================"
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "Python:   $(python --version)"
echo "PyTorch:  $(python -c 'import torch; print(torch.__version__)')"
echo "CUDAtorch:$(python -c 'import torch; print(torch.version.cuda)')"
echo "============================================"

# =============================================================================
# 2) 前置检查 1：必须能用到 GPU（上次失败的根因之一）
# =============================================================================
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("\n[错误] CUDA 不可用 —— 作业落在了 CPU 节点。", file=sys.stderr)
    print("请确认 sbatch 脚本里 --partition=GPU1 与 --gres=gpu:1 已生效，", file=sys.stderr)
    print("或运行 sinfo -o '%P %G' 查看真实分区/GPU 资源名后重交。", file=sys.stderr)
    sys.exit(1)
print(f"GPU 就绪: {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB)")
PY

# =============================================================================
# 3) 前置检查 2：数据目录存在
#    labeled/   = 训练 + 验证集（含 .json 标注）
#    unlabeled/ = 测试集（仅图片，无 .json；只能评估疾病三分类）
# =============================================================================
LABELED_DIR=/workspace/labeled
UNLABELED_DIR=/workspace/unlabeled
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
# 4) 前置检查 3：预训练权重是否已缓存（决定是否加 --offline）
# =============================================================================
CACHE_DIR=~/.cache/torch/hub/checkpoints
CACHE_FILE="$CACHE_DIR/convnext_tiny-983f1562.pth"
OFFLINE=""
if [ -f "$CACHE_FILE" ]; then
    OFFLINE="--offline"
    echo "预训练权重已缓存，使用 --offline（计算节点不联网）。"
else
    echo "[警告] 未找到缓存权重: $CACHE_FILE"
    echo "       建议先在登录节点运行: python download_weights.py --backbone convnext_tiny"
    echo "       本次将尝试联网加载；若计算节点无外网，会退化为随机初始化。"
fi

# =============================================================================
# Step 1：多任务训练（检测 box+annotation + 疾病分类）
# 训练完成后 checkpoint 保存在 outputs/checkpoints/best.pth
# =============================================================================
echo ""
echo ">>> Step 1: 多任务训练（$BACKBONE + $DETECTION_HEAD），$EPOCHS epochs @ ${IMAGE_SIZE}px"
# --test_size 0：labeled/ 全部用于 train/val（不额外切 test），
# 测试集固定用 unlabeled/（Step 3 评估）。
python smoke_test_heads.py  
python train_labeled.py --augmentation tile --detection_head "$DETECTION_HEAD" --image_size "$IMAGE_SIZE" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --backbone "$BACKBONE" --test_size 0 $OFFLINE --resume outputs/checkpoints/last.pth
cp outputs/checkpoints/best.pth outputs/checkpoints/compare_detector_fcos_best.pth

# =============================================================================
# Step 1b：检测分支评估 + 对 unlabeled/ 784 张图片打框标注
# -----------------------------------------------------------------------------
# 训练完成后立即评估检测 box mAP（labeled val 划分），并对 unlabeled/ 全部
# 图片输出三份产物：LabelMe JSON + 可视化 JPG + CSV 汇总。
# 依赖 evaluate_detection.py / annotate_unlabeled.py（已随检测代码一起上传）。
# =============================================================================
echo ""
echo ">>> Step 1b: 检测 mAP 评估（labeled val 划分）"
python evaluate_detection.py --tile --checkpoint compare_detector_fcos_best.pth

echo ""
echo ">>> Step 1c: 对 unlabeled/ 784 张图片打框 + 标注皮损类型"
python annotate_unlabeled.py --checkpoint compare_detector_fcos_best.pth --unlabeled_root "$UNLABELED_DIR" --conf_threshold 0.3

# =============================================================================
# Step 2：皮损统计 → 疾病 分类器（秒级，无需 GPU）
# checkpoint: outputs/checkpoints/annotation_cls_best.pth
# 同时输出"皮损类型 → 疾病"关联矩阵: outputs/annotation_stat_association.json
# =============================================================================
echo ""
echo ">>> Step 2: 训练皮损统计 → 疾病分类器"
python train_annotation_classifier.py

# =============================================================================
# Step 3：测试集评估（unlabeled/）+ 端到端推理示例
# -----------------------------------------------------------------------------
# unlabeled/ 无 .json 标注，故只能评估疾病三分类（无法评估检测分支）。
# 结果写入 outputs/evaluation_unlabeled.json / evaluation_unlabeled_predictions.csv
# =============================================================================
echo ""
echo ">>> Step 3: 测试集评估（unlabeled/，疾病三分类）"
python evaluate_unlabeled.py --checkpoint compare_detector_fcos_best.pth --unlabeled_root "$UNLABELED_DIR"

echo ""
echo ">>> 端到端推理示例（对一张测试图自动标注 + 分类）:"
python -m inference.auto_annotate --image_dir "$UNLABELED_DIR"/predict/ --conf_threshold 0.3

echo ""
echo "============================================"
echo "全部完成 @ $(date)"
echo "产物:"
echo "  outputs/checkpoints/best.pth                多任务模型（检测 + 疾病分类）"
echo "  outputs/checkpoints/annotation_cls_best.pth 皮损统计 → 疾病分类器"
echo "  outputs/annotation_stat_association.json    皮损类型 → 疾病关联矩阵"
echo "  outputs/evaluation_unlabeled.json           unlabeled 测试集评估（疾病三分类）"
echo "  outputs/evaluation_unlabeled_predictions.csv  逐图预测结果"
echo "============================================"
