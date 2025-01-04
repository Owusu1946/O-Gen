from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from src.config import PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME

def initialize_vector_store(embeddings):
    """
    Initialize Pinecone vector store with medical embeddings
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if it doesn't exist
    active_indexes = pc.list_indexes()
    if PINECONE_INDEX_NAME not in [index.name for index in active_indexes]:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    
    return LangchainPinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace="medical_data",
        text_key="text"
    ) 