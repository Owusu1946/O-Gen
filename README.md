# OptiMediX AI Assistant

OptiMediX is a medical AI assistant designed to provide users with accurate and well-reasoned medical information. This chatbot can be trained on custom datasets, allowing it to analyze user queries, retrieve relevant information, and engage in interactive conversations to help users understand medical conditions, treatment options, and healthcare information.

## Features

- **Custom Dataset Training**: Upload your own medical datasets, and the chatbot will use this information to answer questions and provide insights.
- **Interactive Chat Interface**: Users can ask medical questions and receive detailed responses.
- **Follow-up Questions**: The chatbot can ask clarifying questions based on user input, demonstrating human-like critical thinking to arrive at robust answers.
- **Chat History**: Users can view and export their chat history for future reference.

## Technologies Used

- Python
- Streamlit
- LangChain
- Natural Language Processing (NLP)
- Medical Document Processing
- Voyage
- Pinecone
- Generative AI

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/OptiMediX.git
   cd OptiMediX
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).
2. Upload your custom medical dataset using the file uploader in the sidebar.
3. Ask your medical questions in the chat interface.
4. Review the chatbot's responses and follow-up questions for a more interactive experience.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## About

OptiMediX AI Assistant is designed for informational purposes only and should not replace professional medical advice. Always consult a healthcare provider for medical concerns.