from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os

class DocumentProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_split(self):
        # Determine file type based on extension
        _, file_extension = os.path.splitext(self.file_path)
        file_extension = file_extension.lower()

        # Choose appropriate loader based on file extension
        if file_extension == '.pdf':
            loader = PyPDFLoader(self.file_path)
        elif file_extension == '.txt':
            loader = TextLoader(self.file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Only .txt and .pdf files are supported.")

        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(documents)
        return texts

if __name__ == "__main__":
    processor = DocumentProcessor("data/sample_text.txt")
    texts = processor.load_and_split()
    print(f"Processed {len(texts)} text chunks")
