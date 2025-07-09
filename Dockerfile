# 使用官方 PyTorch 2.6.0+ 镜像作为基础镜像
FROM bitnami/pytorch:2.6.0-debian-12-r0

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/workspace/GPT-SoVITS"

# 切换到root用户进行系统包安装
USER root

# 更换为阿里云镜像源并安装系统依赖
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get clean && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace/GPT-SoVITS

# 复制项目文件
COPY . /workspace/GPT-SoVITS/

# 安装 Python 依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 创建必要的目录
RUN mkdir -p output/CanKao output/tts_results

# 暴露端口
EXPOSE 8765 8766

# 设置启动命令
CMD ["python", "websocket_tts_server.py"]