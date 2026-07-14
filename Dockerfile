FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy

WORKDIR /app

# ── system deps (SDL2 for pygame, build tools for pymunk) ─────────────────

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsdl2-dev \
        libsdl2-image-dev \
        libsdl2-mixer-dev \
        libsdl2-ttf-dev \
        libfreetype6-dev \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# ── python deps (cached unless requirements.txt changes) ──────────────────

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── project source ────────────────────────────────────────────────────────

COPY . .

# ── non-root user ─────────────────────────────────────────────────────────

RUN groupadd --gid 1000 agent && \
    useradd --uid 1000 --gid agent --create-home agent && \
    mkdir -p models logs plots media && \
    chown -R agent:agent /app

USER agent

ENTRYPOINT ["python"]
CMD ["train.py"]
