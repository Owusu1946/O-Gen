import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MedicalDocumentLoader:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def load_documents(self) -> List:
        """
        Load medical documents from the specified directory
        """
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)
            raise Warning(f"Created empty directory at {self.directory_path}. Please add PDF documents.")
            
        loader = DirectoryLoader(
            self.directory_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        
        if not documents:
            raise ValueError(f"No PDF documents found in {self.directory_path}")
            
        return self.text_splitter.split_documents(documents) 