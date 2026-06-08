LINT_PATHS = src tests examples benchmarks

.PHONY: install test lint format-check format typecheck demo reproduce benchmark-smoke quality clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q

lint:
	ruff check $(LINT_PATHS)

format-check:
	ruff format --check $(LINT_PATHS)

format:
	ruff check --fix $(LINT_PATHS)
	ruff format $(LINT_PATHS)

typecheck:
	mypy src/green_sarc

quality: lint format-check typecheck test

demo:
	python examples/standalone_agent_loop/run_demo.py

reproduce:
	python -m benchmarks.reproduce

benchmark-smoke:
	python -m pytest -q tests/test_benchmark_smoke.py

clean:
	rm -rf artifacts dist build .pytest_cache .mypy_cache .ruff_cache *.egg-info
