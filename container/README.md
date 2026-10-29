# 环境容器化（conda-pack 快照 → Docker 镜像）

把 HPC 上能跑通的 `dermatology_classifier` conda 环境原样打包成 Docker 镜像，
基底用 Ubuntu 22.04（glibc 2.35），彻底绕开 CentOS 7 的 glibc 2.17 限制。

## 第 1 步：在 HPC 登录节点打快照

```bash
source /phadcloud/home/software/miniforge/etc/profile.d/conda.sh
conda activate base
conda install -c conda-forge conda-pack -y

# 打包环境（会提示忽略个别 pip 包，属正常）
conda pack -n dermatology_classifier -o dermatology_classifier.tar.gz
```

`dermatology_classifier.tar.gz` 就是环境本体（含 torch 等，通常几个 GB）。

## 第 2 步：拷到有 Docker 的服务器

```bash
scp dermatology_classifier.tar.gz user@docker-server:/path/to/container/
```

把本目录的 `Dockerfile`、`build_and_run.sh` 也一起放到 `/path/to/container/`。

## 第 3 步：Docker 服务器上构建并运行

```bash
cd /path/to/container
bash build_and_run.sh
```

手动等价命令：

```bash
# 构建
docker build -t dermatology_env:latest .

# 验证（应打印 torch 版本 + cuda True）
docker run --rm dermatology_env:latest \
  bash -lc "source /opt/conda/bin/activate && \
            python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"

# 跑训练
docker run --gpus all -it --rm \
  -v /path/to/Dermatology_classification:/workspace \
  dermatology_env:latest \
  bash -lc "source /opt/conda/bin/activate && \
            cd /workspace/dermatology_detection && \
            python train_labeled.py --image_size 512 --epochs 50 --batch_size 8"
```

## 关键前提

1. **GPU 直通**：Docker 服务器必须装 NVIDIA 驱动 + `nvidia-container-toolkit`，
   否则 `--gpus all` 会报错、`torch.cuda.is_available()` 为 False。验证：

   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

2. **数据挂载**：训练数据 `labeled/`、`unlabeled/` 不打进镜像，通过 `-v` 挂载进容器。

3. **CUDA 版本**：容器里 torch 用的是环境里自带的 CUDA 运行时；只要宿主机驱动版本
   足够新即可。若报 `libcuda.so.1 not found`，把 Dockerfile 的 `FROM ubuntu:22.04`
   换成 `FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04`（版本按需调）。

## 可选：把镜像推到镜像仓库（方便集群/别人拉取）

```bash
docker tag dermatology_env:latest your-registry/dermatology_env:latest
docker push your-registry/dermatology_env:latest
```
