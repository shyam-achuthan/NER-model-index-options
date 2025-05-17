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
COPY *.py ./

# Copy training data file
COPY training_data.json ./
RUN if [ ! -f "training_data.json" ]; then echo "[]" > training_data.json; fi

# Create necessary directories with proper permissions
RUN mkdir -p models/spacy && \
    chmod -R 777 models

# Expose port
EXPOSE 8000

# Set environment variable to control mode (train, serve, or both)
ENV MODE=both

# Entry point script
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Run the entry script
ENTRYPOINT ["./entrypoint.sh"]