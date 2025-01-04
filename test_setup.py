import os
from dotenv import load_dotenv
from pinecone import Pinecone
from voyageai import Client
import google.generativeai as genai

def test_pinecone():
    print("\nTesting Pinecone connection...")
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        # List all indexes
        active_indexes = pc.list_indexes()
        print("✅ Pinecone connection successful")
        print(f"Available indexes: {active_indexes}")
    except Exception as e:
        print(f"❌ Pinecone error: {str(e)}")

def test_voyage():
    print("\nTesting Voyage AI connection...")
    try:
        client = Client(api_key=os.getenv("VOYAGE_API_KEY"))
        # Test embedding generation
        embeddings = client.embed(texts=["Test medical text"], model="voyage-large-2")
        print("✅ Voyage AI connection successful")
    except Exception as e:
        print(f"❌ Voyage AI error: {str(e)}")

def test_gemini():
    print("\nTesting Google AI connection...")
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Hello!")
        print("✅ Google AI connection successful")
    except Exception as e:
        print(f"❌ Google AI error: {str(e)}")

def main():
    print("Testing API Connections...")
    load_dotenv()
    
    test_pinecone()
    test_voyage()
    test_gemini()

if __name__ == "__main__":
    main() 