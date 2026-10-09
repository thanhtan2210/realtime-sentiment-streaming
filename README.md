# Real-time Sentiment Analysis Pipeline

This project implements a high-performance, industrial-grade end-to-end sentiment analysis pipeline designed for real-time social media monitoring. It leverages a modern Big Data stack and follows advanced Data Engineering principles, including modular architecture, strict data validation, and vectorized processing.

## Business Value and Use Cases

In a digital-first world, understanding public sentiment in real-time provides a significant competitive advantage. This pipeline is designed to solve several critical business challenges:

1. **Brand Reputation Management**: Instantly detect shifts in public sentiment towards a brand or product, allowing for immediate crisis intervention.
2. **Customer Experience Optimization**: Identify dissatisfied customers in real-time to prioritize support tickets and improve Customer Satisfaction (CSAT) scores.
3. **Market Trend Analysis**: Monitor the reception of new product launches or marketing campaigns as they happen across different regions and platforms.
4. **Automated Moderation**: Flag high-intensity negative emotions (e.g., anger, bitterness) for human review or automated filtering.

## Technical Architecture

The system is built with a decoupled, modular architecture to ensure maximum scalability and fault tolerance.

### 1. Ingestion Layer (src/ingestion)
The entry point of the system is a Kafka Producer that simulates high-frequency data streams. 
- **Data Governance**: Utilizes Pydantic for runtime schema validation. Any record failing validation (e.g., missing text, invalid timestamp) is logged and skipped, preventing "poison pills" from entering the pipeline.
- **Resilience**: Implements robust error handling to ensure continuous streaming even when encountering malformed source data.

### 2. Stream Processing Layer (src/processing)
The core logic resides in an Apache Spark Streaming application.
- **Optimization (Vectorized Inference)**: Traditional Spark UDFs suffer from significant overhead due to row-by-row serialization between the JVM and Python. This project utilizes **Pandas UDFs (Vectorized UDFs)** which use **Apache Arrow** for zero-copy data transfer. This allows the model to process data in batches (vectors), utilizing NumPy-level speeds.
- **State Management**: Implements Spark Checkpointing to ensure the pipeline can recover its state and offset position in the event of a system failure.

### 3. Storage Layer
- **Database**: MongoDB serves as the sink for the analytical results, providing a flexible schema for unstructured social media data and supporting rapid aggregation queries.

### 4. Visualization Layer (src/dashboard)
- **Live Monitoring**: A Streamlit dashboard provides a real-time view of the data flowing through the system.
- **Insights**: Visualizes sentiment distribution, geographical activity trends, and platform-specific engagement metrics.

### 5. Model (src/ml)
- **Model Pipeline**: Uses a scikit-learn pipeline (TF-IDF + Logistic Regression) that classifies a tweet as negative, neutral or positive.

## Dataset

[Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment): tweets sent to major US airlines in February 2015, each labelled negative, neutral or positive by Crowdflower (Figure Eight) contributors and published in their "Data for Everyone" library.

- **File**: `data/raw/Tweets.csv`, downloaded by `scripts/fetch_data.py` from a public Hugging Face mirror of the Kaggle dataset (no login needed). The exact URL, download date, row count and sha256 are in [data/raw/SOURCE.md](data/raw/SOURCE.md).
- **License**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial, share-alike). The copy in this repo is redistributed under the same license.

## Project Structure

- src/common: Shared utilities, centralized configuration, and structured logging.
- scripts: Data download script.
- src/ingestion: Data ingestion logic and Pydantic schema definitions.
- src/processing: Spark Streaming jobs and vectorized inference logic.
- src/dashboard: Streamlit visualization application.
- src/ml: Model training and evaluation scripts.
- data/raw: Raw dataset and its source record.
- models: Serialized model artifacts and performance metrics.
- checkpoints: Spark Streaming recovery metadata.

## Setup and Installation

### Prerequisites
- Python 3.10 or higher
- Docker and Docker Compose
- Java Runtime Environment (JRE) 17+

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Infrastructure Deployment
Launch the core services (Kafka, Zookeeper, MongoDB, Spark):
```bash
docker-compose up -d
```

### 3. Model Training
Train the model:
```bash
make train
```

## Running the Pipeline

To run the full end-to-end pipeline, execute these commands in separate terminal sessions:

1. **Ingestion**: Start streaming data from the source to Kafka.
   ```bash
   make producer
   ```
2. **Processing**: Start the Spark Streaming engine.
   ```bash
   make processor
   ```
3. **Dashboard**: Launch the real-time monitoring tool.
   ```bash
   make dashboard
   ```

## Production Considerations

- **Scalability**: The modular design allows Kafka brokers and Spark workers to be scaled horizontally.
- **Fault Tolerance**: Spark Checkpointing and Kafka's distributed nature ensure no data loss during infrastructure outages.
- **Monitoring**: Structured logging is implemented across all modules to facilitate debugging and observability in production environments.
