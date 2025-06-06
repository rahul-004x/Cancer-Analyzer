FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

COPY main.py ./
COPY utils/ ./utils/
COPY attached_assets/ ./attached_assets/

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]