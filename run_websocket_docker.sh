#!/bin/bash

# GPT-SoVITS WebSocket TTS Docker 运行脚本
# 要求 PyTorch 2.6.0+

echo "=== GPT-SoVITS WebSocket TTS Docker 启动脚本 ==="

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker 未安装，请先安装 Docker"
    exit 1
fi

# 检查 docker-compose 是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "错误: docker-compose 未安装，请先安装 docker-compose"
    exit 1
fi

# 检查 NVIDIA Docker 支持
if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "警告: NVIDIA Docker 支持未检测到，将使用 CPU 模式"
    echo "如需 GPU 支持，请安装 nvidia-docker2"
fi

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p output/CanKao output/tts_results

# 检查参考文件
if [ ! -f "output/CanKao/CanKao.wav" ]; then
    echo "警告: 参考音频文件 output/CanKao/CanKao.wav 不存在"
    echo "请将参考音频文件放置到 output/CanKao/CanKao.wav"
fi

if [ ! -f "output/CanKao/CanKao_text.txt" ]; then
    echo "警告: 参考文本文件 output/CanKao/CanKao_text.txt 不存在"
    echo "请将参考文本文件放置到 output/CanKao/CanKao_text.txt"
fi

# 构建镜像
echo "构建 Docker 镜像..."
docker-compose -f docker-compose.websocket.yml build

if [ $? -ne 0 ]; then
    echo "错误: Docker 镜像构建失败"
    exit 1
fi

# 运行容器
echo "启动 WebSocket TTS 服务..."
docker-compose -f docker-compose.websocket.yml up -d

if [ $? -eq 0 ]; then
    echo "✅ WebSocket TTS 服务启动成功！"
    echo "📡 WebSocket 端口: 8765"
    echo "🌐 HTTP 下载端口: 8766"
    echo "📁 输出目录: ./output/tts_results"
    echo ""
    echo "查看日志: docker-compose -f docker-compose.websocket.yml logs -f"
    echo "停止服务: docker-compose -f docker-compose.websocket.yml down"
else
    echo "❌ WebSocket TTS 服务启动失败"
    exit 1
fi 