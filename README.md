# Options Trading NER API

A containerized solution for extracting structured information from natural language options trading queries using Named Entity Recognition (NER).

![Options NER Demo](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbzM2N2c1MHZzY3ZiOHpkZjQxNDdtc2pnNndqbXQ5b3Zoc29lc3lsYyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l0ErWnkLjegNB5LlC/giphy.gif)

## 📋 Overview

The Options Trading NER API extracts key information from options trading queries and returns it in a structured format. It can identify:

- **Index names** (NIFTY50, BANKNIFTY, FINNIFTY, MIDCAPNIFTY)
- **Strike prices** (numeric values like 18500, 25600)
- **Option types** (CE/call, PE/put)

### Example Input/Output

**Input query:**

```
"What's the trend of nifty 23600 call option?"
```

**Structured output:**

```json
{
  "index": "NIFTY50",
  "strikePrice": 23600,
  "strikeType": "CE"
}
```

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose
- Basic terminal/command-line knowledge

### Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/options-ner-api.git
   cd options-ner-api
   ```

2. **Build and run with Docker Compose**

   ```bash
   docker-compose up -d
   ```

3. **Test the API**
   ```bash
   curl -X POST "http://localhost:8000/extract" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the price of nifty 18000 call option?"}'
   ```

## 🏗️ Project Structure

```
options-ner-api/
├── api.py                     # FastAPI application
├── train_model.py             # Model training script
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose configuration
├── requirements.txt           # Python dependencies
├── training_data.json         # Training examples for model
├── entrypoint.sh              # Container startup script
├── test_api.py                # API testing utility
└── models/                    # Directory for trained models
    └── spacy/                 # SpaCy model files
        ├── options_ner_model/ # Trained SpaCy model
        ├── index_mapper.pkl   # Index name mapping
        └── option_mapper.pkl  # Option type mapping
```

## 🛠️ API Reference

### Endpoints

#### Health Check

```
GET /health
```

Returns the current health status of the API and whether the model is loaded.

#### Extract Entities

```
POST /extract
```

Extracts options trading entities from a natural language query.

**Request Body:**

```json
{
  "query": "What is the price of nifty 18000 call option?"
}
```

**Response:**

```json
{
  "index": "NIFTY50",
  "strikePrice": 18000,
  "strikeType": "CE"
}
```

## 🧠 Training the Model

The model can be trained with custom data by:

1. **Updating training_data.json** with examples in the format:

   ```json
   [
     {
       "text": "What's the price of nifty 18500 call?",
       "entities": [[20, 25, "INDEX"], [26, 31, "STRIKE_PRICE"], [32, 36, "OPTION_TYPE"]]
     },
     ...
   ]
   ```

2. **Set training mode** in docker-compose.yml:

   ```yaml
   environment:
     - MODE=train
   ```

3. **Restart the container**:

   ```bash
   docker-compose down
   docker-compose up -d
   ```

4. **Monitor training progress**:
   ```bash
   docker-compose logs -f
   ```

## 🔍 Entity Recognition Details

The API recognizes three entity types:

1. **INDEX**: Stock indices like NIFTY50, BANKNIFTY

   - Variations: "nifty", "bank nifty", "finnifty", "midcap nifty"
   - Standardized to: "NIFTY50", "BANKNIFTY", "FINNIFTY", "MIDCAPNIFTY"

2. **STRIKE_PRICE**: Numeric option strike prices

   - Typically 4-5 digit numbers (15000-45000)

3. **OPTION_TYPE**: Option contract types
   - Call variations: "ce", "call", "call option"
   - Put variations: "pe", "put", "put option"
   - Standardized to: "CE", "PE"

The model handles various query patterns including:

- Standard: "nifty 18000 call"
- Reversed: "18000 ce of nifty"
- With context: "i am holding nifty ce option for 25600"

## 🔧 Configuration Options

The API behavior can be controlled through environment variables in docker-compose.yml:

| Variable  | Description              | Options              |
| --------- | ------------------------ | -------------------- |
| MODE      | Container operation mode | train, serve, both   |
| LOG_LEVEL | Logging verbosity        | DEBUG, INFO, WARNING |

## 📊 Performance Considerations

- **Memory Usage**: ~300-500MB RAM
- **Response Time**: Typically <100ms per query
- **Training Time**: ~3-5 minutes for 300 examples

## 🤝 Contributing

Contributions are welcome! To improve the model:

1. Add more diverse training examples to `training_data.json`
2. Modify training parameters in `train_model.py`
3. Submit a pull request with your enhancements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👏 Acknowledgements

- [SpaCy](https://spacy.io/) for the NER implementation
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
