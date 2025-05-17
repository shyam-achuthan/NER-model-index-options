#!/bin/bash
# Entrypoint script for container startup

set -e  # Exit on error

# Fix directory permissions to ensure writability
fix_permissions() {
  echo "Setting correct permissions on model directories..."
  mkdir -p /app/models/spacy
  chmod -R 777 /app/models
  ls -la /app/models
}

# Check if models already exist
check_models() {
  if [ -d "/app/models/spacy/options_ner_model" ] && [ -f "/app/models/spacy/index_mapper.pkl" ] && [ -f "/app/models/spacy/option_mapper.pkl" ]; then
    echo "Models found at /app/models/spacy/"
    ls -la /app/models/spacy/
    return 0
  else
    echo "Models not found or incomplete in /app/models/spacy/"
    ls -la /app/models/
    return 1
  fi
}

# Function to train the model
train_model() {
  echo "Starting model training..."
  
  # Ensure directories exist with proper permissions
  fix_permissions
  
  # Run the training script
  python /app/train_model.py
  
  # Verify files were created
  if [ -d "/app/models/spacy/options_ner_model" ]; then
    echo "Model training completed successfully."
    ls -la /app/models/spacy/options_ner_model/
  else
    echo "WARNING: Model directory not found after training."
    ls -la /app/models/spacy/
  fi
}

# Function to start the API server
start_server() {
  echo "Starting API server..."
  uvicorn api:app --host 0.0.0.0 --port 8000
}

# Main execution flow
case $MODE in
  train)
    echo "MODE=train: Will train model only and exit"
    train_model
    echo "Model training completed"
    ;;
    
  serve)
    echo "MODE=serve: Will serve existing model"
    if check_models; then
      start_server
    else
      echo "Error: Models not found. Please train the model first or mount models volume."
      echo "You can:"
      echo "1. Set MODE=train to train a new model"
      echo "2. Set MODE=both to train and then serve"
      echo "3. Mount pre-trained models to /app/models/spacy/"
      exit 1
    fi
    ;;
    
  both)
    echo "MODE=both: Will train model then start server"
    train_model
    start_server
    ;;
    
  *)
    echo "Invalid MODE: $MODE"
    echo "Please set MODE to one of: train, serve, both"
    exit 1
    ;;
esac