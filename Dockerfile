FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG APT_MIRROR=http://mirrors.aliyun.com/ubuntu
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn \
    HF_HOME=/workspace/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /workspace/Lance

RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s#http://archive.ubuntu.com/ubuntu#$APT_MIRROR#g" /etc/apt/sources.list && \
        sed -i "s#http://security.ubuntu.com/ubuntu#$APT_MIRROR#g" /etc/apt/sources.list; \
    fi \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip setuptools wheel

COPY requirements.txt ./

RUN python3 -m pip install \
        torch==2.8.0 \
        torchvision==0.23.0 \
        torchaudio==2.8.0 \
        --index-url ${TORCH_INDEX_URL} \
    && python3 -m pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python3", "lance_gradio.py", "--server-name", "0.0.0.0", "--server-port", "7860"]
