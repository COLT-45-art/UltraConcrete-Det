FROM python:3.10-slim

# 系统级依赖（opencv-python-headless 需要这些库才能 import cv2）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/studio/PROJECT

# 先装 CPU 版 torch（避免拉到 CUDA 版，几百 MB 变 2GB+）
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.2.2 \
        torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu

# 再装其余依赖
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 复制项目代码
COPY . /home/studio/PROJECT

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV ULTRASONIC_OUTPUT_ROOT=/tmp/ultrasonic_outputs
ENV INFERENCE_DEVICE=cpu

EXPOSE 7860

CMD ["uvicorn", "stage10_01_fastapi:app", "--host", "0.0.0.0", "--port", "7860"]