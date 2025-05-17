"""
API server for Options NER Model using SpaCy
This script provides an HTTP API to extract index, strike price, and option type from user queries
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import json
import os
import re
import spacy
import logging
from typing import Dict, Optional, List, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define API models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    index: Optional[str] = None
    strikePrice: Optional[int] = None
    strikeType: Optional[str] = None
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# Initialize FastAPI app - this is the variable uvicorn is looking for
app = FastAPI(
    title="Options NER API",
    description="API for extracting options trading entities from natural language queries",
    version="1.0.0"
)

# Global variables to store model
spacy_model = None
index_mapper = {}
option_mapper = {}
model_loaded = False

"""
Debug version of load_model function with extensive logging
Add this to your api.py to diagnose model loading issues
"""

def load_model():
    """Load SpaCy NER model and mappers"""
    global spacy_model, index_mapper, option_mapper, model_loaded
    
    try:
        # Load SpaCy model
        model_path = os.path.join("models", "spacy", "options_ner_model")
        logger.info(f"Attempting to load model from: {model_path}")
        
        if os.path.exists(model_path):
            # Check what's in the model directory
            logger.info(f"Model directory exists. Contents:")
            for item in os.listdir(model_path):
                logger.info(f"  - {item}")
            
            # Try to load the model
            logger.info("Loading SpaCy model...")
            spacy_model = spacy.load(model_path)
            
            # Check model pipeline
            logger.info("Model pipeline components:")
            for name, pipe in spacy_model.pipeline:
                logger.info(f"  - {name}: {type(pipe).__name__}")
            
            # Check if NER component exists
            if "ner" in spacy_model.pipe_names:
                ner = spacy_model.get_pipe("ner")
                logger.info(f"NER labels: {ner.labels}")
            else:
                logger.warning("No NER component found in model!")
                
            logger.info("SpaCy model loaded successfully")
        else:
            logger.warning(f"SpaCy model not found at {model_path}")
            return False
        
        # Load index mapper
        index_mapper_path = os.path.join("models", "spacy", "index_mapper.pkl")
        logger.info(f"Attempting to load index mapper from: {index_mapper_path}")
        if os.path.exists(index_mapper_path):
            with open(index_mapper_path, "rb") as f:
                index_mapper = pickle.load(f)
            logger.info(f"Index mapper loaded successfully with {len(index_mapper)} entries:")
            for k, v in list(index_mapper.items())[:5]:
                logger.info(f"  - '{k}' -> '{v}'")
            if len(index_mapper) > 5:
                logger.info(f"  - ... and {len(index_mapper)-5} more")
        else:
            logger.warning(f"Index mapper not found at {index_mapper_path}")
            return False
        
        # Load option mapper
        option_mapper_path = os.path.join("models", "spacy", "option_mapper.pkl")
        logger.info(f"Attempting to load option mapper from: {option_mapper_path}")
        if os.path.exists(option_mapper_path):
            with open(option_mapper_path, "rb") as f:
                option_mapper = pickle.load(f)
            logger.info(f"Option mapper loaded successfully with {len(option_mapper)} entries:")
            for k, v in list(option_mapper.items())[:5]:
                logger.info(f"  - '{k}' -> '{v}'")
            if len(option_mapper) > 5:
                logger.info(f"  - ... and {len(option_mapper)-5} more")
        else:
            logger.warning(f"Option mapper not found at {option_mapper_path}")
            return False
        
        # Test model with a simple query
        logger.info("Testing model with a simple query...")
        test_query = "nifty 18000 call"
        doc = spacy_model(test_query)
        logger.info(f"Test query: '{test_query}'")
        logger.info(f"Entities found: {[(ent.text, ent.label_) for ent in doc.ents]}")
        
        model_loaded = True
        logger.info("All model components loaded successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

"""
Improved extract_options_data function to handle problematic entity recognition
Replace this in your api.py file
"""

def extract_options_data(query: str) -> Dict[str, Any]:
    """Extract structured data from query using SpaCy model with improved index handling"""
    global spacy_model, index_mapper, option_mapper
    
    # Process the query with SpaCy
    doc = spacy_model(query)
    
    # Initialize result
    result = {
        "index": None,
        "strikePrice": None,
        "strikeType": None
    }
    
    # Log the original query and tokenization for debugging
    logger.info(f"QUERY: '{query}'")
    tokens = [(token.text, token.idx) for token in doc]
    logger.info(f"TOKENS: {tokens}")
    
    # Extract entities from NER model
    entities_found = [(ent.text.lower(), ent.label_) for ent in doc.ents]
    logger.info(f"NER entities found: {entities_found}")
    
    # Clean and process entities
    for ent in doc.ents:
        if ent.label_ == "INDEX":
            # Extract just the index name without additional tokens
            # This handles cases like "sensex sd" -> "sensex"
            index_text = ent.text.lower()
            
            # Check if this contains a known index as a substring
            for known_index in index_mapper.keys():
                if known_index in index_text:
                    logger.info(f"Found known index '{known_index}' within '{index_text}'")
                    index_text = known_index
                    break
            
            # Map to standardized form
            if index_text in index_mapper:
                result["index"] = index_mapper[index_text]
                logger.info(f"INDEX mapped: {index_text} -> {result['index']}")
            else:
                # Try to find any known index within the text
                logger.info(f"Index '{index_text}' not in mapper, trying fallbacks")
                result["index"] = ent.text.upper()
                
        elif ent.label_ == "STRIKE_PRICE":
            try:
                result["strikePrice"] = int(ent.text)
            except ValueError:
                numbers = re.findall(r'\d+', ent.text)
                if numbers:
                    result["strikePrice"] = int(''.join(numbers))
                    
        elif ent.label_ == "OPTION_TYPE":
            option_text = ent.text.lower()
            if option_text in option_mapper:
                result["strikeType"] = option_mapper[option_text]
            else:
                result["strikeType"] = ent.text.upper()
    
    # Enhanced fallbacks for when NER fails to identify entities correctly
    if None in result.values():
        logger.info("Using fallbacks for missing entities")
        
        # Query text in lowercase for easier processing
        query_lower = query.lower()
        
        # 1. Fallback for index - find any known index by name
        if result["index"] is None:
            # First, check for specific indices with higher priority
            priority_indices = {
                "banknifty": "BANKNIFTY",
                "bank nifty": "BANKNIFTY", 
                "finnifty": "FINNIFTY",
                "fin nifty": "FINNIFTY",
                "midcap": "MIDCAPNIFTY",
                "midcap nifty": "MIDCAPNIFTY",
                "sensex": "SENSEX"
            }
            
            # Check for more specific indices first
            index_found = False
            for index_name, canonical in priority_indices.items():
                if index_name in query_lower:
                    result["index"] = canonical
                    logger.info(f"PRIORITY INDEX: Found index '{index_name}' -> '{canonical}'")
                    index_found = True
                    break
            
            # Only check for "nifty" if no other index was found
            if not index_found:
                for index_name, canonical in index_mapper.items():
                    if index_name in query_lower:
                        result["index"] = canonical
                        logger.info(f"FALLBACK: Found index '{index_name}' -> '{canonical}'")
                        break
        
        # 2. Fallback for strike price - find any 4-5 digit number
        if result["strikePrice"] is None:
            strike_match = re.search(r'\b(\d{4,5})\b', query)
            if strike_match:
                result["strikePrice"] = int(strike_match.group(1))
                logger.info(f"FALLBACK: Found strike price {result['strikePrice']}")
        
        # 3. Fallback for option type
        if result["strikeType"] is None:
            # Check option mapper keys
            for option_name, canonical in option_mapper.items():
                if option_name in query_lower:
                    result["strikeType"] = canonical
                    logger.info(f"FALLBACK: Found option type '{option_name}' -> '{canonical}'")
                    break
            
            # If still not found, check for common patterns
            if result["strikeType"] is None:
                if "call" in query_lower or " ce" in query_lower or "ce " in query_lower or " ce " in query_lower:
                    result["strikeType"] = "CE"
                    logger.info("FALLBACK: Found option type 'CE' via pattern")
                elif "put" in query_lower or " pe" in query_lower or "pe " in query_lower or " pe " in query_lower:
                    result["strikeType"] = "PE"
                    logger.info("FALLBACK: Found option type 'PE' via pattern")
    
    # Special handling for known problematic patterns
    if result["index"] and " " in result["index"]:
        # Likely incorrectly joined with another word - use fallbacks
        logger.info(f"Detected problematic index with space: {result['index']}")
        
        # Prioritize more specific indices over general ones
        priority_indices = {
            "banknifty": "BANKNIFTY",
            "bank nifty": "BANKNIFTY", 
            "finnifty": "FINNIFTY",
            "fin nifty": "FINNIFTY",
            "midcap": "MIDCAPNIFTY",
            "midcap nifty": "MIDCAPNIFTY",
            "sensex": "SENSEX"
        }
        
        # Check for more specific indices first
        index_found = False
        for index_name, canonical in priority_indices.items():
            if index_name in query_lower:
                result["index"] = canonical
                logger.info(f"CLEANUP: Fixed index to '{canonical}'")
                index_found = True
                break
                
        # Only if no specific index was found, try the general indices
        if not index_found:
            for index_name, canonical in index_mapper.items():
                if index_name in query_lower:
                    result["index"] = canonical
                    logger.info(f"CLEANUP: Fixed index to '{canonical}'")
                    break
    
    # Final check - if the query contains "banknifty" or "finnifty" but the result is NIFTY50, correct it
    query_lower = query.lower()
    if result["index"] == "NIFTY50":
        if "banknifty" in query_lower or "bank nifty" in query_lower:
            result["index"] = "BANKNIFTY"
            logger.info("FINAL CHECK: Corrected NIFTY50 to BANKNIFTY")
        elif "finnifty" in query_lower or "fin nifty" in query_lower:
            result["index"] = "FINNIFTY"
            logger.info("FINAL CHECK: Corrected NIFTY50 to FINNIFTY")
        elif "midcap" in query_lower:
            result["index"] = "MIDCAPNIFTY"
            logger.info("FINAL CHECK: Corrected NIFTY50 to MIDCAPNIFTY")
        elif "sensex" in query_lower:
            result["index"] = "SENSEX"
            logger.info("FINAL CHECK: Corrected NIFTY50 to SENSEX")
    
    logger.info(f"Final result: {result}")
    return result

def similar_pattern(query: str) -> bool:
    """Check if query matches a pattern like 'what is the trend of [index] [strike] [option]'"""
    # Common patterns that should work with specific format
    patterns = [
        r'(?:what|how).*(?:trend|analysis|status).*(?:nifty|banknifty|finnifty|midcap).*\d{4,5}.*(?:call|put|ce|pe)',
        r'(?:trend|analysis|status).*(?:nifty|banknifty|finnifty|midcap).*\d{4,5}.*(?:call|put|ce|pe)',
        r'.*(?:nifty|banknifty|finnifty|midcap).*\d{4,5}.*(?:call|put|ce|pe).*(?:trend|analysis|status)',
    ]
    
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Loading model...")
    
    # Try to load model
    if load_model():
        logger.info("Model loaded successfully")
    else:
        logger.warning("Failed to load model")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded
    )

@app.post("/extract", response_model=QueryResponse)
async def extract_entities(request: QueryRequest):
    """Extract options trading entities from query"""
    query = request.query
    
    logger.info(f"Received query: {query}")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if not model_loaded:
        # Try to load model if not loaded
        if not load_model():
            raise HTTPException(status_code=503, detail="Model not available")
    
    result = extract_options_data(query)
    logger.info(f"Extracted result: {result}")
    
    return QueryResponse(**result)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Options NER API",
        "version": "1.0.0",
        "description": "Extract structured data from options trading queries using SpaCy",
        "endpoints": {
            "/extract": "POST - Extract entities from query",
            "/health": "GET - Check API health"
        },
        "model_loaded": model_loaded
    }

# This is for local testing only, not used when running with uvicorn in Docker
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)