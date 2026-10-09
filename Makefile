PYTHON := python

.PHONY: help train producer processor dashboard compose-up

help:
	@echo "Available targets:"
	@echo "  make train       - Train model"
	@echo "  make producer    - Replay the test tweets into Kafka (stack must be up)"
	@echo "  make processor   - Run Spark processor"
	@echo "  make dashboard   - Run Streamlit"
	@echo "  make compose-up  - Start infra via docker-compose"

train:
	$(PYTHON) -m src.ml.train

producer:
	docker compose run --rm producer

processor:
	$(PYTHON) -m src.processing.processor

dashboard:
	$(PYTHON) -m streamlit run src/dashboard/app.py

compose-up:
	docker-compose up --build
