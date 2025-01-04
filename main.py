from src.embeddings import get_medical_embeddings
from src.document_loader import MedicalDocumentLoader
from src.vector_store import initialize_vector_store
from src.chatbot import MedicalChatbot

def main():
    # Initialize embeddings
    embeddings = get_medical_embeddings()
    
    # Load and process medical documents
    loader = MedicalDocumentLoader("data/medical_docs")
    documents = loader.load_documents()
    
    # Initialize vector store
    vector_store = initialize_vector_store(embeddings)
    
    # Add documents to vector store
    vector_store.add_documents(documents)
    
    # Initialize chatbot
    chatbot = MedicalChatbot(vector_store)
    
    # Interactive chat loop
    print("Medical Chatbot initialized. Type 'quit' to exit.")
    while True:
        query = input("You: ")
        if query.lower() == 'quit':
            break
        
        response = chatbot.chat(query)
        print(f"Doctor Bot: {response}")

if __name__ == "__main__":
    main() 