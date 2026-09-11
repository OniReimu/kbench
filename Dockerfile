# K-Bench reproduction image. Freezes the CUDA + Python + library stack used for
# the paper's open-weight experiments (single NVIDIA H100 80GB; see docs/COMPUTE.md).
#
# Build (default = main Llama-3.1-8B / Qwen3.5-9B stack):
#   docker build -t kbench .
# For Mistral-7B-v0.3, override the pin:
#   docker build -t kbench:mistral --build-arg PINNED=cross_model_pinned_requirements.txt .
#
# Run (GPU + a mounted host dir so model downloads and results persist):
#   docker run --gpus all -it \
#     -v "$PWD/results:/workspace/kbench/results" \
#     -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
#     kbench bash reproduce.sh topology

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible installs (pulled from the official uv image)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace/kbench

# --- pinned dependency layer (Docker-cached unless a pin file changes) ---
# These pin files fix the bug-prone libs (torch / transformers / peft / accelerate);
# see docs/COMPUTE.md for the model-to-environment mapping.
ARG PINNED=main_pinned_requirements.txt
COPY pyproject.toml main_pinned_requirements.txt qwen_pinned_requirements.txt cross_model_pinned_requirements.txt ./
RUN uv venv --python 3.11 /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install -r "${PINNED}"
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# --- project code + the chcons package (deps already pinned above) ---
COPY . .
RUN uv pip install -e . --no-deps

CMD ["bash", "reproduce.sh", "help"]
