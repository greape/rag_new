from ingestion_repository import (
    clone_github_repo,
    classify_local_files,
    generate_vectors_for_repo,
    process_embeddings,
    add_data_chroma,
    query_chroma,
    test_embedding_generation
)
from models import get_graphcode_embedding,query_emebdding
import os
import logging
import sys

def main():
    try:
        logger = logging.getLogger(__name__)
        # Get user input
        github_url = input("Enter repo URL: ")
        local_path = input("Path to save repo: ")
        
        # Clone the GitHub repository
        clone_github_repo(github_url, local_path)
        print("Successfully cloned the repository.")
        
        # Classify local files
        categorized_files = classify_local_files(local_path)
        print("Successfully categorized files.")
        
        # Generate vectors for the categorized files
        test_file = "new_two\private_gpt\settings\settings.py"  # The file where it's stopping
        if test_embedding_generation(test_file):
            # Proceed with full processing
            vectors = generate_vectors_for_repo(categorized_files)
        else:
            logger.error("Embedding generation test failed - please check the model and dependencies")

        vectors = generate_vectors_for_repo(categorized_files)
        print("Successfully vectorized files.")
        
        # Process embeddings from vectors
        processed_data = process_embeddings(vectors)
        print("Successfully processed embeddings.")
        
        # Add processed data to Chroma 
        add_data_chroma(processed_data)
        print("Successfully added data to Chroma.")

        #query to chroma_db
        question = input("enter your question: ")
        question = query_emebdding(question)
        query_chroma(question)

        # # Check results
        # print("Code Results:", code_results)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()