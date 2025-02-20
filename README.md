# OptiMediX AI Assistant

OptiMediX is a medical AI assistant designed to provide users with accurate and well-reasoned medical information. This chatbot can be trained on custom datasets, allowing it to analyze user queries, retrieve relevant information, and engage in interactive conversations to help users understand medical conditions, treatment options, and healthcare information.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [About](#about)
- [Developer Information](#developer-information)

## Features

| Feature                        | Description                                                                                     |
|--------------------------------|-------------------------------------------------------------------------------------------------|
| Custom Dataset Training        | Upload your own medical datasets, and the chatbot will use this information to answer questions and provide insights. |
| Interactive Chat Interface     | Users can ask medical questions and receive detailed responses.                                 |
| Follow-up Questions            | The chatbot can ask clarifying questions based on user input, demonstrating human-like critical thinking to arrive at robust answers. |
| Chat History                   | Users can view and export their chat history for future reference.                             |
| Multi-language Support         | The chatbot can be trained to understand and respond in multiple languages, making it accessible to a wider audience. |
| Contextual Learning            | The AI continuously learns from user interactions, improving its responses over time.          |

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
   git clone https://github.com/Owusu1946/O-Gen.git
   cd O-Gen
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

Contributions are welcome! If you have suggestions for improvements or new features, please follow the guidelines below:

### How to Contribute

1. **Fork the Repository**: Click on the "Fork" button at the top right of the repository page to create your own copy of the repository.

2. **Clone Your Fork**: Clone your forked repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/OptiMediX.git
   ```

3. **Create a New Branch**: Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**: Implement your changes and ensure that your code adheres to the project's coding standards.

5. **Commit Your Changes**: Commit your changes with a descriptive message:
   ```bash
   git commit -m "Add a new feature or fix a bug"
   ```

6. **Push to Your Fork**: Push your changes to your forked repository:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**: Go to the original repository and click on "New Pull Request". Select your branch and submit the pull request.

### Contribution Guidelines

| Guideline                     | Description                                                                                     |
|-------------------------------|-------------------------------------------------------------------------------------------------|
| Code Quality                  | Ensure your code is clean, well-documented, and follows the project's coding standards.        |
| Testing                       | Include tests for any new features or bug fixes.                                             |
| Documentation                 | Update the documentation to reflect any changes made to the codebase.                        |
| Issue Tracking                | If you find a bug or have a feature request, please open an issue in the repository.          |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## About

OptiMediX AI Assistant is designed for informational purposes only and should not replace professional medical advice. Always consult a healthcare provider for medical concerns.

## Developer Information

**Owusu Kenneth**  
Email: [owusukenneth77@gmail.com](mailto:owusukenneth77@gmail.com)  
Phone: +233559182794  

As the developer of OptiMediX, I am passionate about leveraging technology to improve healthcare accessibility and understanding. I welcome feedback and collaboration to enhance this project further. Feel free to reach out with any questions or suggestions!