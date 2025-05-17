#!/bin/bash
# Script to prepare SpaCy model and build Docker image

# Create model directories
mkdir -p models/spacy

# Check if SpaCy model exists
if [ -d "options_ner_model" ]; then
    echo "Copying SpaCy model..."
    cp -r options_ner_model models/spacy/
    cp index_mapper.pkl models/spacy/
    cp option_mapper.pkl models/spacy/
    echo "SpaCy model copied successfully"
else
    echo "SpaCy model directory not found"
    exit 1
fi

# Build Docker image
echo "Building Docker image..."
docker build -t options-ner-api:latest .

echo "Docker image built successfully"
echo "To run the container: docker run -p 8000:8000 options-ner-api:latest"
echo "Or use docker-compose: docker-compose up -d"