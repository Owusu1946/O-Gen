from voyageai import Client
import os
from dotenv import load_dotenv

load_dotenv()

def test_voyage_connection():
    client = Client(api_key=os.getenv("VOYAGE_API_KEY"))
    try:
        # Test a simple embedding
        result = client.embed(texts=["Test medical text"], model="voyage-2")
        print("Voyage AI connection successful!")
        print(f"Embedding dimension: {len(result[0])}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_voyage_connection() 