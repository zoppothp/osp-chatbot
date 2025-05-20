import sys
import os
import importlib

def test_huggingface_hub():
    print("Testing huggingface_hub integration...")
    try:
        # Try to import huggingface_hub
        import huggingface_hub
        print(f"Using huggingface_hub version: {huggingface_hub.__version__}")
        
        # Check if cached_download is available
        if hasattr(huggingface_hub, 'cached_download'):
            print("cached_download function is available in huggingface_hub")
            return True
        else:
            print("ERROR: cached_download function is NOT available in huggingface_hub")
            print("Available attributes:", dir(huggingface_hub))
            return False
    except ImportError as e:
        print(f"Error importing huggingface_hub: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

def test_sentence_transformers():
    print("\nTesting sentence_transformers integration...")
    try:
        # Try to import sentence_transformers
        import sentence_transformers
        print(f"Using sentence_transformers version: {sentence_transformers.__version__}")
        return True
    except ImportError as e:
        print(f"Error importing sentence_transformers: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

def test_embedding_indexer():
    print("\nTesting EmbeddingIndexer initialization...")
    try:
        from embedding_indexer import EmbeddingIndexer
        indexer = EmbeddingIndexer()
        print("EmbeddingIndexer initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing EmbeddingIndexer: {str(e)}")
        return False

if __name__ == "__main__":
    hub_success = test_huggingface_hub()
    st_success = test_sentence_transformers()
    indexer_success = test_embedding_indexer()
    
    if hub_success and st_success and indexer_success:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)