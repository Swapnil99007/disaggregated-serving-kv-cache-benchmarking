# kvbench image: router, simulated workers, and the benchmark client.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# [gpu] adds nvidia-ml-py for NVML telemetry; it degrades gracefully on hosts without a GPU
RUN pip install ".[gpu]"
COPY configs ./configs

EXPOSE 8000 9000
ENTRYPOINT ["kvbench"]
CMD ["--help"]
