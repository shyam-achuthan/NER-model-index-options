import requests
import json
import time
import argparse

def test_api(url):
    """Test the Options NER API with example queries"""
    
    # Sample queries to test
    test_queries = [
        "What's the price of nifty 18500 call?",
        "How is bank nifty 40000 pe doing today?",
        "Give me information on finnifty put option 20000",
        "Should I buy midcap nifty 30000 CE?",
        "nifty 17500 pe premium",
        "What's the latest on sensex 60000 pe options?",
        "I want to see nifty 19000 call option chain",
        "Price trend of banknifty 35000 put",
        "Should I hold or sell my fin nifty 25000 calls?",
        "Update on midcap 28000 ce"
    ]
    
    # Test health endpoint
    print("Testing health endpoint...")
    try:
        health_response = requests.get(f"{url}/health")
        print(f"Health Status: {health_response.status_code}")
        print(json.dumps(health_response.json(), indent=2))
        print("-" * 50)
    except requests.exceptions.RequestException as e:
        print(f"Error checking health: {e}")
        return
    
    # Process each query
    print("Testing entity extraction...")
    print("-" * 50)
    
    results = []
    
    for query in test_queries:
        print(f"Query: {query}")
        
        try:
            # Start timing
            start_time = time.time()
            
            # Make API request
            response = requests.post(
                f"{url}/extract",
                json={"query": query}
            )
            
            # End timing
            elapsed_time = time.time() - start_time
            
            # Process response
            if response.status_code == 200:
                result = response.json()
                print(f"Result: {json.dumps(result, indent=2)}")
                print(f"Time: {elapsed_time:.4f} seconds")
                
                # Store result for summary
                results.append({
                    "query": query,
                    "result": result,
                    "time": elapsed_time
                })
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            
        print("-" * 50)
    
    # Print summary
    if results:
        print("\nSummary:")
        total_time = sum(r["time"] for r in results)
        avg_time = total_time / len(results)
        print(f"Processed {len(results)} queries")
        print(f"Average processing time: {avg_time:.4f} seconds")
        
        success_count = sum(1 for r in results if all(r["result"].values()))
        success_rate = success_count / len(results) * 100
        print(f"Success rate: {success_rate:.1f}% ({success_count}/{len(results)} queries fully extracted)")

def interactive_mode(url):
    """Interactive mode to test the API with user input"""
    print("Interactive Options NER API testing")
    print("Enter 'q' or 'quit' to exit")
    print("-" * 50)
    
    while True:
        # Get query from user
        query = input("Enter your query: ")
        
        if query.lower() in ('q', 'quit', 'exit'):
            break
            
        if not query:
            continue
            
        try:
            # Start timing
            start_time = time.time()
            
            # Make API request
            response = requests.post(
                f"{url}/extract",
                json={"query": query}
            )
            
            # End timing
            elapsed_time = time.time() - start_time
            
            # Process response
            if response.status_code == 200:
                result = response.json()
                print(f"Result: {json.dumps(result, indent=2)}")
                print(f"Time: {elapsed_time:.4f} seconds")
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Test Options NER API")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.url)
    else:
        test_api(args.url)

if __name__ == "__main__":
    main()
 200:
                result = response.json()
                print(f"Result: {json.dumps(result, indent=2)}")
                print(f"Time: {elapsed_time:.4f} seconds")
                
                # Store result for summary
                results.append({
                    "query": query,
                    "result": result,
                    "time": elapsed_time
                })
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            
        print("-" * 50)
    
    # Print summary
    if results:
        print("\nSummary:")
        total_time = sum(r["time"] for r in results)
        avg_time = total_time / len(results)
        print(f"Processed {len(results)} queries")
        print(f"Average processing time: {avg_time:.4f} seconds")
        
        success_count = sum(1 for r in results if all(r["result"].values()))
        success_rate = success_count / len(results) * 100
        print(f"Success rate: {success_rate:.1f}% ({success_count}/{len(results)} queries fully extracted)")

def interactive_mode(url, model_type="spacy"):
    """Interactive mode to test the API with user input"""
    print(f"Interactive Options NER API testing (model: {model_type})")
    print("Enter 'q' or 'quit' to exit")
    print("-" * 50)
    
    while True:
        # Get query from user
        query = input("Enter your query: ")
        
        if query.lower() in ('q', 'quit', 'exit'):
            break
            
        if not query:
            continue
            
        try:
            # Start timing
            start_time = time.time()
            
            # Make API request
            response = requests.post(
                f"{url}/extract",
                json={"query": query, "model_type": model_type}
            )
            
            # End timing
            elapsed_time = time.time() - start_time
            
            # Process response
            if response.status_code == 200:
                result = response.json()
                print(f"Result: {json.dumps(result, indent=2)}")
                print(f"Time: {elapsed_time:.4f} seconds")
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Test Options NER API")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--model", default="spacy", choices=["spacy", "bert"], help="Model type")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.url, args.model)
    else:
        test_api(args.url, args.model)

if __name__ == "__main__":
    main()