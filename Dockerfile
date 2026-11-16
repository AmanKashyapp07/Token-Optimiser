FROM node:20-slim AS frontend-builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml ./backend/
RUN pip install --no-cache-dir ./backend

COPY backend/ ./backend/
COPY --from=frontend-builder /app/dist ./dist

EXPOSE 8000
CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
