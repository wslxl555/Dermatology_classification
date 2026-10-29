#!/bin/bash
# =============================================================================
# 一键搬迁脚本 —— 服务器到期前，把「代码 + 数据 + 权重 + conda 环境 + Docker 镜像」
# 全部打包到同一个目录，之后 scp 回本地即可。
#
# 在【服务器】上运行：  bash migrate_server.sh
# 产物：               ~/migration_YYYYmmdd_HHMMSS/
# =============================================================================
set -e

# ===== ★ 按实际路径修改这几行 =====
PROJECT_ROOT=~/Dermatology_classification_3/Dermatology_classification   # 项目根目录
CONDA_ENV=dermatology_classifier                                          # conda 环境名
CONDA_SH=/phadcloud/home/software/miniforge/etc/profile.d/conda.sh        # conda 初始化脚本
DOCKER_IMAGE=dermatology_env:latest                                        # 已构建的 Docker 镜像（没有就留空 ""）
# ===================================

# ---- 预检 ----
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "[错误] 项目目录不存在: $PROJECT_ROOT"
    echo "       请改脚本开头的 PROJECT_ROOT 为实际路径。"
    exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=~/migration_$STAMP
mkdir -p "$OUT"
echo "=============================================="
echo " 搬迁输出目录: $OUT"
echo "=============================================="

step() { echo; echo ">>> $1"; }

# ===================== [1/5] 项目（代码 + 数据 + 权重）=====================
step "[1/5] 打包项目（代码 + 数据 + checkpoint）..."
tar -czf "$OUT/project.tar.gz" \
    -C "$PROJECT_ROOT" \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='*/site-packages' \
    --exclude='*/Lib' \
    --exclude='*/Scripts' \
    .

# ===================== [2/5] 预训练权重缓存 =====================
step "[2/5] 打包预训练权重缓存 ..."
if [ -d ~/.cache/torch/hub/checkpoints ]; then
    tar -czf "$OUT/torch_hub_cache.tar.gz" -C ~/.cache/torch/hub/checkpoints
else
    echo "  [跳过] 无 ~/.cache/torch/hub/checkpoints"
fi
if [ -d ~/.cache/huggingface ]; then
    tar -czf "$OUT/hf_cache.tar.gz" -C ~/.cache/huggingface 2>/dev/null || true
fi

# ===================== [3/5] conda 环境快照 =====================
step "[3/5] conda-pack 环境快照 ..."
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate base
    conda install -c conda-forge conda-pack -y
    conda pack -n "$CONDA_ENV" -o "$OUT/env_${CONDA_ENV}.tar.gz"
else
    echo "  [警告] 找不到 $CONDA_SH，跳过环境打包"
fi

# ===================== [4/5] Docker 镜像 =====================
step "[4/5] 导出 Docker 镜像 ..."
if [ -n "$DOCKER_IMAGE" ] && command -v docker >/dev/null 2>&1; then
    if docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
        SAFE_NAME=$(echo "$DOCKER_IMAGE" | tr ':/' '__')
        docker save -o "$OUT/image_${SAFE_NAME}.tar" "$DOCKER_IMAGE"
    else
        echo "  [警告] 镜像 $DOCKER_IMAGE 不存在，跳过（先 docker build 再跑本脚本）"
    fi
else
    echo "  [跳过] 未指定镜像，或本机没有 docker"
fi

# ===================== [5/5] 恢复说明 + 清单 =====================
step "[5/5] 生成清单与恢复说明 ..."
cat > "$OUT/RESTORE.md" <<EOF
# 搬迁恢复说明

## 到新服务器后恢复

### 1) 恢复 conda 环境
\`\`\`bash
mkdir -p ~/miniforge/envs/$CONDA_ENV
tar -xzf env_${CONDA_ENV}.tar.gz -C ~/miniforge/envs/$CONDA_ENV
source ~/miniforge/envs/$CONDA_ENV/bin/activate
\`\`\`

### 2) 恢复 Docker 镜像（若有 image_*.tar）
\`\`\`bash
docker load -i image_*.tar
\`\`\`

### 3) 恢复项目代码 + 数据
\`\`\`bash
mkdir -p /path/to/Dermatology_classification
tar -xzf project.tar.gz -C /path/to/Dermatology_classification
\`\`\`

### 4) 恢复预训练权重缓存
\`\`\`bash
mkdir -p ~/.cache/torch/hub
tar -xzf torch_hub_cache.tar.gz -C ~/.cache/torch/hub
\`\`\`
EOF

du -sh "$OUT"/* > "$OUT/MANIFEST.txt"
echo "================ 搬迁完成 ================"
cat "$OUT/MANIFEST.txt"
echo "=========================================="
echo
echo "下载到本地（在本地 Git Bash / WSL 里执行）："
echo "  scp -r $(whoami)@$(hostname):$OUT ."
