.DEFAULT_GOAL := help
.PHONY: help setup app eval test lint format clean

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install deps (creates .venv via uv)
	uv sync

app: ## Run the Streamlit demo
	uv run streamlit run app.py

eval: ## Run the eval loop (lands W3)
	@echo "Eval loop lands in W3 — see evals/README.md for planned layout."
	@exit 1

test: ## Run pytest
	uv run pytest

lint: ## Ruff check + format check
	uv run ruff check .
	uv run ruff format --check .

format: ## Ruff auto-format
	uv run ruff format .
	uv run ruff check --fix .

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache **/__pycache__
