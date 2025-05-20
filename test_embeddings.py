import sys
import os
from embedding_indexer import EmbeddingIndexer

def test_embedding_indexer():
    print("Testing EmbeddingIndexer initialization...")
    try:
        indexer = EmbeddingIndexer()
        print("EmbeddingIndexer initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing EmbeddingIndexer: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_embedding_indexer()
    if success:
        print("Test passed!")
        sys.exit(0)
    else:
        print("Test failed!")
        sys.exit(1)