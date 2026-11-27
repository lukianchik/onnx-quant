# Build once with Internet, run the published model and fixed dataset offline.
FROM python:3.13-slim AS demo-builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false
WORKDIR /build
RUN python -m pip install --no-cache-dir 'torch==2.15.0+cpu' \
    --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir \
    'onnx==1.23.2' 'transformers==5.24.0' \
    'datasets==5.2.0' 'numpy==2.5.5'
COPY scripts/prepare_demo.py ./scripts/prepare_demo.py
RUN python scripts/prepare_demo.py

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HOME=/app/data/hf-cache \
    QUANTLAB_EXAMPLES=/app/examples \
    TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY quantlab ./quantlab
COPY app.py ./app.py
COPY assets ./assets
COPY tests ./tests
COPY --from=demo-builder /build/examples/ /app/examples/
RUN useradd -m -u 1000 -s /usr/sbin/nologin appuser && \
    mkdir -p /app/data /app/examples && \
    chown appuser:appuser /app/data && \
    test -s /app/examples/bert_tiny_fp32.onnx && \
    test -s /app/examples/evaluation_samples.json
USER appuser
EXPOSE 7860
CMD ["python", "app.py"]
