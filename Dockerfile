FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
COPY clawloop/ clawloop/

RUN pip install --no-cache-dir ".[server]"

EXPOSE 8400

CMD ["python", "-m", "clawloop.server", "--host", "0.0.0.0", "--port", "8400"]
