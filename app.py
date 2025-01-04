import streamlit as st
from src.embeddings import get_medical_embeddings
from src.document_loader import MedicalDocumentLoader
from src.vector_store import initialize_vector_store
from src.chatbot import MedicalChatbot
from datetime import datetime
import os

def initialize_chatbot():
    with st.spinner('Initializing medical knowledge base...'):
        embeddings = get_medical_embeddings()
        loader = MedicalDocumentLoader("data/medical_docs")
        documents = loader.load_documents()
        vector_store = initialize_vector_store(embeddings)
        vector_store.add_documents(documents)
        return MedicalChatbot(vector_store)

def format_chat_history(messages):
    history = []
    for msg in messages:
        timestamp = msg.get("timestamp", "")
        formatted_msg = f"**{msg['role'].title()}** ({timestamp}):\n{msg['content']}\n"
        history.append(formatted_msg)
    return "\n".join(history)

def main():
    # Page configuration
    st.set_page_config(
        page_title="OptiMediX AI Assistant",
        page_icon="🏥",
        layout="wide"
    )

    # Initialize session state
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = initialize_chatbot()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_history" not in st.session_state:
        st.session_state.show_history = False
    if "awaiting_follow_up" not in st.session_state:
        st.session_state.awaiting_follow_up = False
        st.session_state.follow_up_questions = []

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/caduceus.png", width=100)
        st.title("OptiMediX AI")
        
        # File uploader for document uploads
        uploaded_files = st.file_uploader("Upload Medical Documents", type=["pdf", "docx"], accept_multiple_files=True)
        
        if st.button("Process Uploaded Documents"):
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    # Save the uploaded file to a temporary location
                    with open(os.path.join("data/medical_docs", uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.success("Documents uploaded and processed successfully.")
                # Reinitialize the chatbot with the new documents
                st.session_state.chatbot = initialize_chatbot()
            else:
                st.warning("Please upload at least one document.")

        # Chat History Controls
        st.markdown("### 📝 Chat History")
        st.session_state.show_history = st.toggle("Show Chat History", st.session_state.show_history)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.follow_up_questions = []
            st.session_state.awaiting_follow_up = False
            st.rerun()
        
        # About Section
        st.markdown("""
        ### About
        I'm your medical AI assistant, trained on professional medical knowledge. 
        I can help you understand:
        - Medical conditions
        - Treatment options
        - Healthcare information
        - Medical terminology
        
        ### Guidelines
        - Provide clear, specific questions
        - Include relevant context
        - For emergencies, contact your healthcare provider
        
        _This AI assistant is for informational purposes only and should not replace professional medical advice._
        """)

    # Main chat interface
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🏥 OptiMediX AI Assistant")
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            Welcome to OptiMediX AI Assistant. I'm here to help you understand medical information better.
            Please note that I provide information based on medical documents, but this should not replace
            professional medical advice.
        </div>
        """, unsafe_allow_html=True)

        # Chat History View
        if st.session_state.show_history and st.session_state.messages:
            with st.expander("Chat History", expanded=True):
                st.markdown("### Previous Conversations")
                st.markdown(format_chat_history(st.session_state.messages))
                if st.button("Export Chat History"):
                    history_text = format_chat_history(st.session_state.messages)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="Download Chat History",
                        data=history_text,
                        file_name=f"medical_chat_history_{timestamp}.txt",
                        mime="text/plain"
                    )

    # Chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🏥"):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask your medical question here..."):
        # User message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        })
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant", avatar="🏥"):
            with st.spinner('Analyzing medical knowledge base...'):
                try:
                    response = st.session_state.chatbot.chat(prompt)
                    
                    # Check if the chatbot is asking for follow-up questions
                    if isinstance(response, list):  # Assuming response is a list of follow-up questions
                        st.session_state.awaiting_follow_up = True
                        st.session_state.follow_up_questions = response
                        follow_up_message = "I found some information related to your query. Can you please clarify: " + " ".join(response)
                        st.markdown(follow_up_message)
                    else:
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                except Exception as e:
                    error_message = f"⚠️ I apologize, but I encountered an error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

    # Handle follow-up questions
    if st.session_state.awaiting_follow_up:
        follow_up_response = st.chat_input("Please provide your response to the follow-up questions...")
        if follow_up_response:
            # Process the follow-up response
            st.session_state.messages.append({
                "role": "user",
                "content": follow_up_response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(follow_up_response)

            # Get the final response from the chatbot based on the follow-up
            with st.chat_message("assistant", avatar="🏥"):
                with st.spinner('Analyzing your response...'):
                    try:
                        final_response = st.session_state.chatbot.chat(follow_up_response)
                        st.markdown(final_response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_response,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        st.session_state.awaiting_follow_up = False  # Reset the follow-up state
                        st.session_state.follow_up_questions = []  # Clear follow-up questions
                    except Exception as e:
                        error_message = f"⚠️ I apologize, but I encountered an error: {str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_message,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

    # Footer
    st.markdown("""
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #f0f2f6;'>
        <small>OptiMediX AI Assistant • For informational purposes only • Not medical advice</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 