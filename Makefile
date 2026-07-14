.PHONY: install train eval render plot test lint docker-build docker-train clean

# ─── Setup ────────────────────────────────────────────────────────────────────

install:
	pip install -e .

# ─── Training & Evaluation ───────────────────────────────────────────────────

train:
	python main.py train

eval:
	python main.py evaluate

render:
	python main.py render

plot:
	python main.py plot

# ─── Quality ─────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

lint:
	black --check .
	isort --check .

format:
	black .
	isort .

# ─── Docker ──────────────────────────────────────────────────────────────────

docker-build:
	docker build -f docker/Dockerfile -t double-pendulum .

docker-train:
	docker compose -f docker/docker-compose.yml up train

# ─── Cleanup ─────────────────────────────────────────────────────────────────

clean:
	rm -rf __pycache__ .pytest_cache *.egg-info build dist
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
