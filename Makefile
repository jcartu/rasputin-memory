.PHONY: test lint typecheck benchmark benchmark-full start stop

test:
	pytest tests/ -q -k "not integration"

lint:
	ruff check .

typecheck:
	cd tools && mypy brain/ pipeline/ --ignore-missing-imports

benchmark:
	python3 benchmarks/run_benchmark.py --check-thresholds

benchmark-full:
	python3 benchmarks/run_benchmark.py --full --check-thresholds

start:
	python3 tools/hybrid_brain.py

stop:
	docker compose down
