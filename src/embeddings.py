from voyageai import Client
from langchain_voyageai import VoyageAIEmbeddings
from src.config import VOYAGE_API_KEY

def get_medical_embeddings():
    """
    Initialize Voyage AI embeddings model specifically trained on medical data
    """
    embeddings = VoyageAIEmbeddings(
        voyage_api_key=VOYAGE_API_KEY,
        model="voyage-large-2",
        show_progress_bar=True
    )
    return embeddings 