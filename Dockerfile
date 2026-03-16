FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
COPY lfx/ lfx/
COPY config/ config/

RUN pip install --no-cache-dir ".[n8n]"

EXPOSE 8400

CMD ["python", "-m", "lfx.server", "--host", "0.0.0.0", "--port", "8400"]