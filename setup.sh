#!/bin/bash
# Script to create a minimal environment for training and serving NER model

set -e  # Exit on error

echo "Creating Options NER API environment..."

# Create project directory
mkdir -p options-ner-api
cd options-ner-api

# Create models directory
mkdir -p models/spacy

# Create required files
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn==0.23.2
pydantic==2.4.2
spacy==3.7.2
scikit-learn==1.3.0
matplotlib==3.7.2
numpy==1.24.3
EOF

# Create empty training_data.json (optional)
echo '[]' > training_data.json

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies and necessary tools for debugging
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get update && \
    apt-get install -y --no-install-recommends procps curl vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy all code files
COPY *.py /app/

# Copy training data if it exists
COPY training_data.json /app/ 2>/dev/null || echo "No training data found, will generate synthetic data"

# Create necessary directories with proper permissions
RUN mkdir -p /app/models/spacy && \
    chmod -R 777 /app/models

# Expose port
EXPOSE 8000

# Set environment variable to control mode (train, serve, or both)
ENV MODE=both

# Entry point script
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Run the entry script
ENTRYPOINT ["/app/entrypoint.sh"]
EOF

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3'

services:
  options-ner-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:rw
      - ./training_data.json:/app/training_data.json:ro
    restart: unless-stopped
    environment:
      - LOG_LEVEL=INFO
      - MODE=both  # Options: train, serve, both
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health || exit 0"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    # Set user to root to avoid permission issues with mounted volumes
    user: root
EOF

# Download the training script, API server, and entrypoint from GitHub Gist
curl -s -o train_model.py "https://gist.githubusercontent.com/your-username/gist-id/raw/train_model.py"
curl -s -o api.py "https://gist.githubusercontent.com/your-username/gist-id/raw/api.py" 
curl -s -o entrypoint.sh "https://gist.githubusercontent.com/your-username/gist-id/raw/entrypoint.sh"
curl -s -o test_api.py "https://gist.githubusercontent.com/your-username/gist-id/raw/test_api.py"

# Make entrypoint executable
chmod +x entrypoint.sh

echo "Environment setup complete!"
echo "You can now run: cd options-ner-api && docker-compose up -d"
echo "The API will be available at: http://localhost:8000"