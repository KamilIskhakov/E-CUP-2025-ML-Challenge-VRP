FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deterministic_vrp_solver /app/deterministic_vrp_solver
COPY run_inference.py /app/run_inference.py

# minimal deps for inference
RUN pip install --no-cache-dir polars pyarrow numpy plotly

ENTRYPOINT ["python", "run_inference.py"]

