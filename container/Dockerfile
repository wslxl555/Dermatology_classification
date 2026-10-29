# =============================================================================
# 皮肤病分类/检测 环境镜像（conda-pack 快照方式）
#
# 基底用 Ubuntu 22.04（glibc 2.35），绕开 CentOS 7 的 glibc 2.17 限制。
# 环境本体来自 conda-pack 打好的 tar 包，直接解包，无需重新安装依赖。
#
# 构建前：把 dermatology_classifier.tar.gz 放到本目录下。
# =============================================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

# OpenCV / Pillow 等需要的少量系统库；其余依赖都在 conda 环境里自包含
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libxrender1 \
        libxext6 \
        libsm6 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# 解包 conda 环境快照到 /opt/conda
COPY dermatology_classifier.tar.gz /tmp/dermatology_classifier.tar.gz
RUN mkdir -p /opt/conda \
    && tar -xzf /tmp/dermatology_classifier.tar.gz -C /opt/conda \
    && rm /tmp/dermatology_classifier.tar.gz

ENV PATH="/opt/conda/bin:${PATH}" \
    CONDA_PREFIX="/opt/conda"

WORKDIR /workspace

# 运行训练时：source /opt/conda/bin/activate && python xxx.py
CMD ["/bin/bash"]
