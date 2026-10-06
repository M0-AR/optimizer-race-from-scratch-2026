FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY experiments/ ./experiments/
COPY data/ ./data/
ENV PYTHONPATH=/app
CMD ["python3", "experiments/run_all.py", "--quick"]
