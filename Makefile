.DEFAULT_GOAL := help

.PHONY: help install dev test check smoke docs docs-serve build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install with dev dependencies
	uv sync --extra dev

dev: ## Install with all extras
	uv sync --all-extras

test: ## Run unit tests
	uv run pytest tests/ -x -q --timeout=60

check: ## Quick import/syntax sanity check
	uv run python -m compileall -q clawloop/
	@echo "Compile check passed"

smoke: ## Run the full smoke-test script
	bash scripts/smoke.sh

docs: ## Build documentation
	uv sync --extra docs
	uv run mkdocs build --strict

docs-serve: ## Serve docs locally (http://127.0.0.1:8000)
	uv sync --extra docs
	uv run mkdocs serve

build: ## Build sdist + wheel
	uv sync --extra release
	uv run python -m build --outdir dist/

clean: ## Remove build artifacts
	rm -rf dist/ build/ site/ *.egg-info .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
