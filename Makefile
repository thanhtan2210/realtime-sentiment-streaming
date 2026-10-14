PYTHON := python

.PHONY: help train test up replay dashboard down

help:
	@echo "Available targets:"
	@echo "  make train    - Train the model and write models/metrics.json"
	@echo "  make test     - Run the tests"
	@echo "  make up       - Build and start Kafka, MongoDB, the Spark processor and the dashboard (localhost:8501)"
	@echo "  make dashboard - Run the dashboard on the recorded snapshot, without Docker"
	@echo "  make replay   - Replay the test tweets into Kafka (stack must be up)"
	@echo "  make down     - Stop the stack and delete its data (Kafka log, MongoDB, checkpoints)"
	@echo ""
	@echo "After changing stream.spark.shuffle_partitions (or SPARK_SHUFFLE_PARTITIONS) run"
	@echo "'make down' before 'make up': the checkpoint keeps the old number of partitions."

train:
	$(PYTHON) -m src.ml.train

test:
	$(PYTHON) -m pytest -q

up:
	docker compose up -d --build

replay:
	docker compose run --rm producer --speedup 600

dashboard:
	$(PYTHON) -m streamlit run app.py

# Also the only way to apply a new stream.spark.shuffle_partitions: the
# checkpoint volume fixes the number of state partitions it was created with.
down:
	docker compose down -v
