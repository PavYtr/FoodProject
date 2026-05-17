FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y \
    git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN git clone --depth 1 https://github.com/ByteDance-Seed/depth-anything-3.git . \
    && rm -rf .git

COPY export_init.py src/depth_anything_3/utils/export/__init__.py
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e . --no-deps

RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='depth-anything/DA3-SMALL', local_dir='./models/DA3-SMALL')"


FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

ENV MODEL_PATH=/app/models/DA3-SMALL
COPY depth_module.py .
COPY get_depth.py .

CMD ["python3", "depth_module.py"]
