# Offline demo for the network attack early-warning world model.
#   docker build -t attack-demo .
#   docker run --rm -p 8501:8501 attack-demo      then open http://localhost:8501
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# libgomp1: OpenMP runtime used by xgboost (listed in requirements.txt)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (the default Linux wheel pulls in CUDA), then the pinned requirements
COPY requirements.txt .
RUN pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "dashboard/demo.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
