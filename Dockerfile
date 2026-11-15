# syntax=docker/dockerfile:1
FROM node:22-bookworm-slim AS web-build
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1 NEXT_OUTPUT=standalone
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

FROM node:22-bookworm-slim AS runtime
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG INSTALL_TENSORFLOW=false

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000 \
    PATH=/opt/venv/bin:$PATH \
    PYTHON_BIN=/opt/venv/bin/python \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    FRUITCOUNT_MODEL_ROOT=/models \
    FRUITCOUNT_TEMP_DIR=/tmp/fruitcount \
    YOLO_CONFIG_DIR=/home/fruitcount/.config/Ultralytics \
    YOLO_AUTOINSTALL=False \
    YOLO_OFFLINE=True \
    HOME=/home/fruitcount

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv libgl1 libglib2.0-0 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
COPY requirements-core.txt requirements.txt /opt/requirements/
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel --index-url https://pypi.org/simple \
    # Download ONLY the selected torch wheels from the CPU/CUDA index.
    # Resolve their normal dependencies on PyPI, not the incomplete torch mirror.
    && python -m pip download --no-cache-dir --no-deps --only-binary=:all: \
         --dest /tmp/torch-wheels "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
         --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install --no-cache-dir /tmp/torch-wheels/*.whl --index-url https://pypi.org/simple \
    && rm -rf /tmp/torch-wheels \
    && if [ "$INSTALL_TENSORFLOW" = "true" ]; then \
         python -m pip install --no-cache-dir -r /opt/requirements/requirements.txt --index-url https://pypi.org/simple; \
       else \
         python -m pip install --no-cache-dir -r /opt/requirements/requirements-core.txt --index-url https://pypi.org/simple; \
       fi \
    && python -m pip check \
    && python -c "import torch, torchvision, cv2, numpy, ultralytics, ensemble_boxes, onnxruntime, h5py"

WORKDIR /app
RUN groupadd --gid 10001 fruitcount \
    && useradd --uid 10001 --gid fruitcount --create-home fruitcount \
    && mkdir -p /models /tmp/fruitcount /app/.next/cache \
    && chown -R fruitcount:fruitcount /app /tmp/fruitcount /home/fruitcount
COPY --from=web-build --chown=fruitcount:fruitcount /app/.next/standalone ./
COPY --from=web-build --chown=fruitcount:fruitcount /app/.next/static ./.next/static
COPY --from=web-build --chown=fruitcount:fruitcount /app/public ./public
# Python files are launched dynamically and must be copied explicitly.
COPY --chown=fruitcount:fruitcount scripts/*.py ./scripts/
COPY --chown=fruitcount:fruitcount docker/healthcheck.mjs ./docker/healthcheck.mjs
USER fruitcount
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 CMD ["node", "docker/healthcheck.mjs"]
CMD ["node", "server.js"]
