"""
Training script for Options NER Model
This script trains a SpaCy NER model to extract index, strike price, and option type from queries
"""

import spacy
import pickle
import json
import re
import os
import random
import logging
from pathlib import Path
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


 
"""
Fix for the load_training_data function in train_model.py
This ensures we generate proper training data from default variations
"""

def load_training_data():
    """Load training data from JSON file if it exists"""
    training_data_path = "training_data.json"
    if os.path.exists(training_data_path):
        try:
            with open(training_data_path, 'r') as f:
                data = json.load(f)
            
            if not data:  # Empty file or empty list
                logger.info("Training data file exists but is empty. Generating synthetic data instead.")
                return create_training_data()
                
            # Convert the loaded data to the format expected by spaCy
            training_data = [(item["text"], {"entities": item["entities"]}) for item in data]
            
            # Define standard variations for indexes and option types
            # These are used to create proper mappers even if training data is provided
            index_variations = {
                "NIFTY50": ["nifty", "nifty 50", "nifty50", "nifty index", "nifty 50 index"],
                "BANKNIFTY": ["bank nifty", "banknifty", "nifty bank", "banking index", "bank index"],
                "FINNIFTY": ["fin nifty", "finnifty", "nifty fin", "financial index", "financial nifty"],
                "MIDCAPNIFTY": ["midcap nifty", "nifty midcap", "midcap", "midcap index", "mid cap nifty"]
            }
            
            option_variations = {
                "CE": ["ce", "call", "call option", "call options", "call option contract"],
                "PE": ["pe", "put", "put option", "put options", "put option contract"]
            }
            
            logger.info(f"Loaded {len(training_data)} examples from {training_data_path}")
            logger.info(f"Using standard index and option variations for mappers")
            return training_data, index_variations, option_variations
        
        except Exception as e:
            logger.error(f"Error loading training data from file: {str(e)}")
            logger.info("Generating synthetic training data instead")
    
    # If file doesn't exist or there was an error, generate synthetic data
    return create_training_data()
"""
Fix for the create_entity_mappers function in train_model.py
This function is responsible for creating the correct mappers during training
"""

def create_entity_mappers(index_variations, option_variations):
    """Create flat mapping dictionaries for standardizing entity values"""
    
    index_mapper = {}
    option_mapper = {}
    
    # Create index mapper
    for canonical, variations in index_variations.items():
        for variant in variations:
            index_mapper[variant.lower()] = canonical
    
    # Create option mapper
    for canonical, variations in option_variations.items():
        for variant in variations:
            option_mapper[variant.lower()] = canonical
    
    # IMPORTANT FIX: Always use these default mappings regardless of variations
    # This ensures we always have proper mappings even if the variations dictionary is empty
    index_mapper = {
        "nifty": "NIFTY50",
        "nifty 50": "NIFTY50",
        "nifty50": "NIFTY50",
        "nifty index": "NIFTY50",
        "nifty 50 index": "NIFTY50",
        
        "bank nifty": "BANKNIFTY",
        "banknifty": "BANKNIFTY",
        "nifty bank": "BANKNIFTY",
        "banking index": "BANKNIFTY",
        "bank index": "BANKNIFTY",
        
        "fin nifty": "FINNIFTY",
        "finnifty": "FINNIFTY",
        "nifty fin": "FINNIFTY",
        "financial index": "FINNIFTY",
        "financial nifty": "FINNIFTY",
        
        "midcap nifty": "MIDCAPNIFTY",
        "nifty midcap": "MIDCAPNIFTY",
        "midcap": "MIDCAPNIFTY",
        "midcap index": "MIDCAPNIFTY",
        "mid cap nifty": "MIDCAPNIFTY"
    }
    
    option_mapper = {
        "ce": "CE",
        "call": "CE",
        "call option": "CE",
        "call options": "CE",
        "call option contract": "CE",
        
        "pe": "PE",
        "put": "PE",
        "put option": "PE",
        "put options": "PE",
        "put option contract": "PE"
    }
    
    return index_mapper, option_mapper
 
"""
Improvements to train_model.py to enhance NER for challenging patterns
"""

"""
Improvements to train_model.py to enhance NER for challenging patterns
"""

def create_training_data():
    """Generate synthetic training data with more emphasis on challenging patterns"""
    
    # Define standard variations for index names - ADDED SENSEX
    index_variations = {
        "NIFTY50": ["nifty", "nifty 50", "nifty50", "nifty index", "nifty 50 index"],
        "BANKNIFTY": ["bank nifty", "banknifty", "nifty bank", "banking index", "bank index"],
        "FINNIFTY": ["fin nifty", "finnifty", "nifty fin", "financial index", "financial nifty"],
        "MIDCAPNIFTY": ["midcap nifty", "nifty midcap", "midcap", "midcap index", "mid cap nifty"],
        "SENSEX": ["sensex", "bse sensex", "sensex index", "bse 30", "sensex 30"]
    }
    
    # Define standard variations for option types
    option_variations = {
        "CE": ["ce", "call", "call option", "call options", "call option contract"],
        "PE": ["pe", "put", "put option", "put options", "put option contract"]
    }
    
    # Create template queries - standard patterns
    standard_templates = [
        "What's the price of {index} {strike} {option_type}?",
        "How is {index} {strike} {option_type} doing today?",
        "Give me information on {index} {option_type} {strike}",
        "Should I buy {index} {strike} {option_type}?",
        "What about {index} {strike} {option_type}?",
        "Check {index} {strike} {option_type} for me",
        "{index} {option_type} {strike} price?",
        "Tell me about {index} {option_type} at strike {strike}",
        "I'm looking at {index} {strike} {option_type}",
        "What's the premium for {index} {strike} {option_type}?",
        "I want to trade {index} {strike} {option_type}",
        "{index} {strike} {option_type} analysis",
        "What's your view on {index} {strike} {option_type}?",
        "Is {index} {strike} {option_type} a good buy?",
        "{index} {option_type} at {strike} looks good?"
    ]
    
    # Create template queries - reversed patterns (IMPORTANT: extra focus on these)
    reversed_templates = [
        "i am holding {strike} {option_type} of {index} should i hold or sell",
        "holding {strike} {option_type} of {index} for a week now",
        "considering {strike} {option_type} of {index} for trading",
        "my portfolio has {strike} {option_type} of {index}",
        "bought {strike} {option_type} of {index} yesterday",
        "trading with {strike} {option_type} of {index} this week",
        "should i exit my {strike} {option_type} of {index}?",
        "what's the outlook for {strike} {option_type} of {index}?",
        "analysis of {strike} {option_type} of {index} position",
        "how much is {strike} {option_type} of {index} worth now?",
        # Add even more variation of these patterns
        "my {strike} {option_type} {index} position is in profit",
        "evaluating {strike} {option_type} {index} options for next week",
        "planning to roll over {strike} {option_type} {index} contract",
        "got {strike} {option_type} {index} at a good price",
        "worried about my {strike} {option_type} {index} trade"
    ]
    
    # Generate training examples
    training_data = []
    strike_prices = list(range(15000, 45000, 500))  # Generate strike prices 
    
    # Generate 300 examples with strong focus on reversed patterns
    standard_count = 180  # 60% standard patterns
    reversed_count = 120  # 40% reversed patterns (extra emphasis)
    
    # Generate standard pattern examples
    for _ in range(standard_count):
        # Randomly select index, strike price and option type
        index_key = random.choice(list(index_variations.keys()))
        index_text = random.choice(index_variations[index_key])
        strike = random.choice(strike_prices)
        option_key = random.choice(list(option_variations.keys()))
        option_text = random.choice(option_variations[option_key])
        
        # Select a template and fill it
        template = random.choice(standard_templates)
        text = template.format(index=index_text, strike=strike, option_type=option_text)
        
        # Create entities
        entities = []
        
        # Find index position
        index_start = text.lower().find(index_text.lower())
        if index_start != -1:
            entities.append((index_start, index_start + len(index_text), "INDEX"))
            
        # Find strike price position
        strike_text = str(strike)
        strike_start = text.find(strike_text)
        if strike_start != -1:
            entities.append((strike_start, strike_start + len(strike_text), "STRIKE_PRICE"))
            
        # Find option type position
        option_start = text.lower().find(option_text.lower())
        if option_start != -1:
            entities.append((option_start, option_start + len(option_text), "OPTION_TYPE"))
            
        # Add to training data
        training_data.append((text, {"entities": entities}))
    
    # Generate reversed pattern examples (extra focus on these)
    for _ in range(reversed_count):
        # Randomly select index, strike price and option type
        index_key = random.choice(list(index_variations.keys()))
        index_text = random.choice(index_variations[index_key])
        strike = random.choice(strike_prices)
        option_key = random.choice(list(option_variations.keys()))
        option_text = random.choice(option_variations[option_key])
        
        # Select a template and fill it
        template = random.choice(reversed_templates)
        text = template.format(index=index_text, strike=strike, option_type=option_text)
        
        # Create entities - carefully find each entity
        entities = []
        
        # Find index position - careful with "of {index}"
        index_patterns = [
            f"of {index_text.lower()}",  # "of nifty"
            f"{index_text.lower()} position",  # "nifty position"
            f"{index_text.lower()} options",  # "nifty options" 
            f"{index_text.lower()} contract",  # "nifty contract"
            f"{index_text.lower()} trade"  # "nifty trade"
        ]
        
        for pattern in index_patterns:
            pattern_pos = text.lower().find(pattern)
            if pattern_pos != -1:
                # Calculate the actual index position by adjusting for "of " or other prefix
                adjust = 0
                if "of " in pattern:
                    adjust = 3  # length of "of "
                
                index_start = pattern_pos + adjust
                entities.append((index_start, index_start + len(index_text), "INDEX"))
                break
        else:
            # Direct search if patterns not found
            index_start = text.lower().find(index_text.lower())
            if index_start != -1:
                entities.append((index_start, index_start + len(index_text), "INDEX"))
            
        # Find strike price position
        strike_text = str(strike)
        strike_start = text.find(strike_text)
        if strike_start != -1:
            entities.append((strike_start, strike_start + len(strike_text), "STRIKE_PRICE"))
            
        # Find option type position
        option_start = text.lower().find(option_text.lower())
        if option_start != -1:
            entities.append((option_start, option_start + len(option_text), "OPTION_TYPE"))
            
        # Add to training data
        training_data.append((text, {"entities": entities}))
    
    logger.info(f"Generated {len(training_data)} training examples")
    logger.info(f"Standard patterns: {standard_count}, Reversed patterns: {reversed_count}")
    logger.info(f"Using standard index and option variations for mappers")
    return training_data, index_variations, option_variations

# Updated parameters for training to improve learning
def train_ner_model(training_data, output_dir, n_iter=50):  # Increased iterations
    """Train a SpaCy NER model with improved parameters"""
    
    # Same code as before but with these changes:
    # 1. Increase iterations (n_iter=50 instead of 30)
    # 2. Lower the drop rate to retain more information 
    # (change drop=0.5 to drop=0.3 in nlp.update())
    # 3. Use a smaller batch size growth rate for more updates
    # (change compounding(4.0, 32.0, 1.001) to compounding(4.0, 16.0, 1.0005))
    
    # Check if output directory exists
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # Load base model
    nlp = spacy.blank("en")
    
    # Create the NER component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Add entity labels
    for _, annotations in training_data:
        for ent in annotations.get("entities", []):
            ner.add_label(ent[2])
    
    # Split into training and evaluation sets
    train_data, eval_data = train_test_split(training_data, test_size=0.2, random_state=42)
    logger.info(f"Training set: {len(train_data)} examples")
    logger.info(f"Evaluation set: {len(eval_data)} examples")
    
    # Start training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    # Only train NER
    with nlp.disable_pipes(*other_pipes):
        # Initialize optimizer
        optimizer = nlp.begin_training()
        
        # Training loop
        losses = {}
        for i in range(n_iter):
            # Shuffle training data
            random.shuffle(train_data)
            losses_iter = {}
            
            # Batch up the examples - smaller growth rate for more updates
            batches = minibatch(train_data, size=compounding(4.0, 16.0, 1.0005))
            for batch in batches:
                # Create list of examples
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                
                # Update the model - reduced dropout for better retention
                nlp.update(
                    examples,
                    drop=0.3,  # Lower dropout rate (was 0.5)
                    sgd=optimizer,
                    losses=losses_iter
                )
            
            # Log progress
            if i % 5 == 0:
                logger.info(f"Iteration {i+1}/{n_iter}, Loss: {losses_iter.get('ner', 0):.3f}")
            
            # Store losses
            losses[i] = losses_iter.get("ner", 0)
    
    # Rest of the function stays the same
    
    # Save model
    nlp.to_disk(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
    return nlp
 
def main():
    """Main function to train and save the model"""
    
    # Debug information
    print(f"Current working directory: {os.getcwd()}")
    print(f"Directory listing: {os.listdir('.')}")
    
    # Create output directory structure with explicit permissions
    output_dir = os.path.join("models", "spacy", "options_ner_model")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check the directory was created successfully
    if os.path.exists(output_dir):
        print(f"Output directory created at: {output_dir}")
    else:
        print(f"Failed to create output directory at: {output_dir}")
        
    # List permissions on models directory
    try:
        print(f"Permissions on models directory: {os.stat('models').st_mode}")
        print(f"Permissions on models/spacy directory: {os.stat(os.path.join('models', 'spacy')).st_mode}")
    except Exception as e:
        print(f"Error checking permissions: {e}")
    
    # Load or generate training data
    logger.info("Preparing training data...")
    training_data, index_variations, option_variations = load_training_data()
    
    # Create entity mappers
    logger.info("Creating entity mappers...")
    index_mapper, option_mapper = create_entity_mappers(index_variations, option_variations)
    
    # Train the model
    logger.info("Training NER model...")
    model = train_ner_model(training_data, output_dir)
    
    # Save mappers
    index_mapper_path = os.path.join("models", "spacy", "index_mapper.pkl")
    with open(index_mapper_path, "wb") as f:
        pickle.dump(index_mapper, f)
    
    option_mapper_path = os.path.join("models", "spacy", "option_mapper.pkl")
    with open(option_mapper_path, "wb") as f:
        pickle.dump(option_mapper, f)
    
    # Verify files were saved correctly
    print("\nVerifying saved files:")
    if os.path.exists(os.path.join(output_dir, "meta.json")):
        print(f"✅ Model saved successfully at {output_dir}")
    else:
        print(f"❌ Model not saved correctly at {output_dir}")
        
    if os.path.exists(index_mapper_path):
        print(f"✅ Index mapper saved at {index_mapper_path}")
    else:
        print(f"❌ Index mapper not saved at {index_mapper_path}")
        
    if os.path.exists(option_mapper_path):
        print(f"✅ Option mapper saved at {option_mapper_path}")
    else:
        print(f"❌ Option mapper not saved at {option_mapper_path}")
        
    # List all files in model directory for debugging
    try:
        print("\nFiles in model directory:")
        model_files = os.listdir(output_dir)
        print(f"Found {len(model_files)} files: {model_files[:5]}...")
    except Exception as e:
        print(f"Error listing model files: {e}")
    
    logger.info("Model and mappers saved successfully!")
    
    # Test the model with a few examples
    logger.info("\nTesting the model with sample queries:")
    test_queries = [
        "What's the price of nifty 18500 call?",
        "How is bank nifty 40000 pe doing today?",
        "Give me information on finnifty put option 20000",
        "what is the trend of nifty 23600 call option"
    ]
    
    for query in test_queries:
        doc = model(query)
        entities = {ent.label_: ent.text for ent in doc.ents}
        logger.info(f"Query: {query}")
        logger.info(f"Entities: {entities}")


if __name__ == "__main__":
    main()