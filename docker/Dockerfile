# ── Stage 1: Base image with system dependencies ─────────────────────────────
FROM python:3.12-slim AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps for Pygame (SDL2) and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev \
    libportmidi-dev \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 2: Install Python dependencies ─────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Copy source and install project ──────────────────────────────────
FROM deps AS app

COPY . .
RUN pip install --no-cache-dir -e .

# Default: headless training (no display)
ENV SDL_VIDEODRIVER=dummy
ENV SDL_AUDIODRIVER=dummy

# Entry point
ENTRYPOINT ["python", "main.py"]
CMD ["train"]
