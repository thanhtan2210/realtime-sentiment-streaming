PYTHON := python

.PHONY: help train test up replay down

help:
	@echo "Available targets:"
	@echo "  make train    - Train the model and write models/metrics.json"
	@echo "  make test     - Run the tests"
	@echo "  make up       - Build and start Kafka, MongoDB and the Spark processor"
	@echo "  make replay   - Replay the test tweets into Kafka (stack must be up)"
	@echo "  make down     - Stop the stack and delete its data"

train:
	$(PYTHON) -m src.ml.train

test:
	$(PYTHON) -m pytest -q

up:
	docker compose up -d --build

replay:
	docker compose run --rm producer --speedup 600

down:
	docker compose down -v
