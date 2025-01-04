from pinecone import Pinecone
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME

def delete_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if PINECONE_INDEX_NAME in [index.name for index in pc.list_indexes()]:
        print(f"Deleting index: {PINECONE_INDEX_NAME}")
        pc.delete_index(PINECONE_INDEX_NAME)
        print("Index deleted successfully")
    else:
        print("Index does not exist")

if __name__ == "__main__":
    delete_pinecone_index() 