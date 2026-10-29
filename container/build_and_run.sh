#!/bin/bash
# =============================================================================
# 构建镜像 + 示例运行
# 用法：把本脚本、Dockerfile、dermatology_classifier.tar.gz 放在同一目录，
#       然后 bash build_and_run.sh
# =============================================================================
set -e

IMAGE=dermatology_env:latest
# ★ 改成 Docker 服务器上项目目录的实际路径（训练时要挂载进容器）
PROJECT_DIR=/root/Dermatology_classification

# 1) 构建镜像
echo ">>> 构建镜像 $IMAGE ..."
docker build -t "$IMAGE" .

# 2) 验证环境（打印 python / torch / CUDA 是否可用）
echo ">>> 验证环境 ..."
docker run --rm "$IMAGE" bash -lc \
    "source /opt/conda/bin/activate && \
     python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())'"

# 3) 跑训练（GPU 直通 + 挂载项目与数据目录）
echo ">>> 运行训练 ..."
docker run --gpus all -it --rm \
    -v "$PROJECT_DIR":/workspace \
    -v /root/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints \
    "$IMAGE" \
    bash -lc "source /opt/conda/bin/activate && \
              cd /workspace/dermatology_detection/script && \
              bash run_hpc_train.sh"
