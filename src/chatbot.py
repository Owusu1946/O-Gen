from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

class MedicalChatbot:
    def __init__(self, vector_store):
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=5  # Keep last 5 conversations
        )
        
        self.initial_prompt = """You are an expert medical AI assistant. Use the following pieces of context to provide accurate, well-reasoned medical information. Always cite your sources and explain your reasoning.
        
        Context: {context}
        
        Chat History: {chat_history}
        Current Question: {question}
        
        Please provide a detailed, accurate response while:
        1. Citing specific information from the provided context
        2. Explaining your reasoning clearly
        3. Mentioning any limitations or uncertainties
        4. Being clear about what is factual (from sources) vs general medical knowledge
        """
        
        self.qa_prompt = PromptTemplate(
            template=self.initial_prompt,
            input_variables=["context", "chat_history", "question"]
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.1,
                convert_system_message_to_human=True,
                max_output_tokens=2048,
            ),
            retriever=vector_store.as_retriever(
                search_kwargs={"k": 8}
            ),
            memory=self.memory,
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt}
        )

    def chat(self, query: str) -> str:
        """
        Process user query and return medical advice
        """
        try:
            # First, get relevant context from the vector store
            context = self.chain.retriever.retrieve(query)
            if not context:
                return "I couldn't find any relevant information. Can you provide more details?"

            # Ask follow-up questions based on the context
            follow_up_questions = self.generate_follow_up_questions(query, context)
            if follow_up_questions:
                return "I found some information related to your query. Can you please clarify: " + " ".join(follow_up_questions)

            # If no follow-up questions, proceed to generate an answer
            response = self.chain.invoke({"question": query})
            sources = [doc.metadata.get('source', 'Unknown') for doc in response.get('source_documents', [])]
            answer = response["answer"]
            
            # Add source attribution if available
            if sources:
                answer += "\n\nSources consulted: " + ", ".join(set(sources))
            
            return answer
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return "I apologize, but I encountered an error processing your query. Please try again."

    def generate_follow_up_questions(self, query, context):
        # Analyze the query and context to generate relevant follow-up questions
        questions = []
        
        # Example logic for generating follow-up questions
        if "pain" in query.lower():
            questions.append("Can you describe the location of the pain?")
            questions.append("How long have you been experiencing this pain?")
            questions.append("Is the pain constant or does it come and go?")
            questions.append("Have you noticed any other symptoms, such as nausea or fever?")
        
        if "cough" in query.lower():
            questions.append("How long have you had the cough?")
            questions.append("Is it dry or productive (producing mucus)?")
            questions.append("Do you have any other symptoms, like fever or shortness of breath?")
        
        # Add more conditions and questions as needed
        return questions 