from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
import importlib.util
import os
import sys

class EmbeddingIndexer:
    def __init__(self):
        # Try to use HuggingFaceEmbeddings with fallback options
        try:
            # First check if sentence_transformers is available
            sentence_transformers_spec = importlib.util.find_spec("sentence_transformers")
            if sentence_transformers_spec is None:
                print("Warning: sentence_transformers package not found. Attempting to install it...")
                # Try to install the package
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers==2.2.2", "torch>=2.0.0", "transformers>=4.6.0", "huggingface-hub==0.16.4"])
                print("Installation completed. Trying to import again...")

            # Try to import it explicitly
            import sentence_transformers
            print(f"Using sentence_transformers version: {sentence_transformers.__version__}")

            # Try to import torch
            import torch
            print(f"Using torch version: {torch.__version__}")

            # Try to import transformers
            import transformers
            print(f"Using transformers version: {transformers.__version__}")

            # Check huggingface_hub version
            try:
                import huggingface_hub
                print(f"Using huggingface_hub version: {huggingface_hub.__version__}")

                # Check if cached_download is available
                if not hasattr(huggingface_hub, 'cached_download'):
                    print("Warning: huggingface_hub does not have cached_download function. Installing compatible version...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub==0.16.4"])
                    print("Installed huggingface-hub==0.16.4. Reloading module...")
                    import importlib
                    importlib.reload(huggingface_hub)
                    print(f"Now using huggingface_hub version: {huggingface_hub.__version__}")
            except ImportError:
                print("Warning: huggingface_hub not found. Installing compatible version...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub==0.16.4"])
                print("Installed huggingface-hub==0.16.4")

            # Initialize HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print("Successfully initialized HuggingFaceEmbeddings")

        except Exception as e:
            print(f"Error initializing HuggingFaceEmbeddings: {str(e)}")
            raise ImportError(
                f"Could not initialize embeddings: {str(e)}. "
                "Please ensure sentence-transformers, torch, and transformers are installed correctly."
            )

    def create_vectorstore(self, texts):
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        return vectorstore

if __name__ == "__main__":
    from document_processor import DocumentProcessor

    processor = DocumentProcessor("data/sample_text.txt")
    texts = processor.load_and_split()

    indexer = EmbeddingIndexer()
    vectorstore = indexer.create_vectorstore(texts)
    print("Vector store created successfully")
