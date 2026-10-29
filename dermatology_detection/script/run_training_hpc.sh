#!/bin/bash
#SBATCH --job-name=test0906 # 任务名称，必填，默认是提交脚本名
#SBATCH --partition=T1 # 分区名称
#SBATCH --nodes=1 # 节点数，必填，否则默认 1 节点
#SBATCH --ntasks-per-node=1 # 每节点任务数，必填，否则默认 1 任务
#SBATCH --cpus-per-task=64 # 每任务 CPU 核数

# =============================================================================
# Skin Lesion Detection & Classification Training
# =============================================================================
#
# 提交前请确认:
#   1. 在登录节点运行:  python download_weights.py
#   2. 确认项目目录正确: cd ~/jiangwei/Dermatology_classification/dermatology_detection
# =============================================================================

# --- 初始化 conda（SLURM 环境必须手动 source）---
#srun -N1 -n32  --nodelist=node202  --pty bash
source /phadcloud/home/software/miniforge/etc/profile.d/conda.sh
conda activate dermatology_classifier

# --- 切换到项目目录 ---
cd ..

# --- 打印环境信息 ---
echo "============================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "TorchVision: $(python -c 'import torchvision; print(torchvision.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "============================================"

# 图像增强
#python -m data.augment_minority --class contact_dermatitis --factor 10

 Step 1: 快速检查（仅分类，3 epoch，无检测分支）
#echo ""
#echo ">>> STEP 1: Sanity check (tiny mode, classification only)"
#python train.py --tiny --offline

 Step 2: 完整训练（50 epoch，FCOS 检测头启用）
echo ""
echo ">>> STEP 2: Full training (50 epochs, detection enabled)"
python train_labeled.py --image_size 512 --epochs 50 --batch_size 8
echo ""
echo "Training completed at $(date)"

# 评估:
#python evaluate.py --checkpoint best.pth --split test
#
# 单张推理:
#   python -m inference.predict --image /path/to/image.jpg --checkpoint best.pth --visualize

