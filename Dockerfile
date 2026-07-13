FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch==2.3.1+cpu" \
    && CMAKE_ARGS="-DGGML_NATIVE=OFF" FORCE_CMAKE=1 \
        python -m pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

ARG EMBEDDING_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
ARG LOCAL_MODEL_REPO=Qwen/Qwen2.5-1.5B-Instruct-GGUF
ARG LOCAL_MODEL_FILE=qwen2.5-1.5b-instruct-q4_k_m.gguf

RUN mkdir -p /models/all-MiniLM-L6-v2 \
    && python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${EMBEDDING_MODEL_ID}', local_dir='/models/all-MiniLM-L6-v2', local_dir_use_symlinks=False, allow_patterns=['*.json','*.txt','*.safetensors','1_Pooling/config.json'])" \
    && python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='${LOCAL_MODEL_REPO}', filename='${LOCAL_MODEL_FILE}', local_dir='/models', local_dir_use_symlinks=False)" \
    && rm -rf /root/.cache/huggingface

ENV EMBEDDING_MODEL_PATH=/models/all-MiniLM-L6-v2
ENV LOCAL_GGUF_PATH=/models/${LOCAL_MODEL_FILE}
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# Local-first submission mode: every category tries the bundled local model,
# and only answers our validators REJECT are escalated to Fireworks (~5.7k
# tokens on the 215-case set vs ~15.7k for classic routing). The pure
# zero-token variant (ALLOW_ESCALATION=0) scored BELOW the 50% accuracy gate
# on the real judge despite passing offline checks -- never re-enable it for
# a submission without a live-judge measurement.
ENV FORCE_ALL_LOCAL=1
ENV ALLOW_ESCALATION=1

COPY agent ./agent
COPY main.py .

CMD ["python", "-m", "main"]
