from models import get_graphcode_embedding,generate_text_embedding
from vector_db import code_collection,docs_collection
import os
import subprocess
import numpy as np
import torch
import traceback
import logging
import traceback
import sys
import gc
import time
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to clone from github repository
def clone_github_repo(github_url, local_path):
    # Check if the directory exists
    if os.path.exists(local_path):
        if os.listdir(local_path):  # Check if directory is not empty
            print(f"Destination path '{local_path}' already exists and is not empty.")
            return False
    else:
        # Create the directory if it does not exist
        os.makedirs(local_path, exist_ok=True)

    try:
        # Clone the repository
        subprocess.run(['git', 'clone', github_url, local_path], check=True)
        print(f"Repository cloned successfully to {local_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Function to classify files and exclude hidden/git-related files
def classify_local_files(repo_path):
    categorized_files = {"code": [], "docs": []}
    exclude_patterns = ['.git', '.sample', 'HEAD']  # Add any unwanted patterns

    # Define file extension mappings
    code_extensions = ['.py', '.js', '.java', '.cpp', '.c', '.h', '.go', '.rb', '.swift', '.kt', '.yaml', '.yml', '.json', '.toml', '.ini', '.cfg']
    docs_extensions = ['.md', '.txt', '.rst', '.doc', '.docx', '.pdf', '.html', '.htm']

    # Walk through all files in the local repo directory
    for root, dirs, files in os.walk(repo_path):
        # Exclude .git folder from the walk
        dirs[:] = [d for d in dirs if d not in ['.git']]
        
        for file_name in files:
            # Skip hidden and unwanted files based on patterns
            if any(pattern in file_name for pattern in exclude_patterns):
                continue
            
            # Get the full path to the file
            file_path = os.path.join(root, file_name)
            
            # Get file extension
            _, ext = os.path.splitext(file_name)
            ext = ext.lower()  # Normalize the extension to lowercase

            # Classify based on extension
            if ext in code_extensions:
                categorized_files["code"].append(file_path)
            elif ext in docs_extensions:
                categorized_files["docs"].append(file_path)

    return categorized_files

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def clear_memory():
    """Clear unused memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
def generate_vectors_for_repo(categorized_files, batch_size=10, sleep_time=2):
    vectors = {"code": [], "docs": []}
    
    def process_batch(file_batch, file_type):
        batch_vectors = []
        for file_path in file_batch:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if not content.strip():
                    logger.warning(f"Empty file: {file_path}")
                    continue
                
                logger.info(f"Generating embedding for: {file_path}")
                
                if file_type == "code":
                    embedding = get_graphcode_embedding(content)
                else:
                    embedding = generate_text_embedding(content)
                
                if embedding is not None:
                    batch_vectors.append((file_path, embedding))
                    logger.info(f"Successfully generated embedding for: {file_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
        
        return batch_vectors

    # Process code files in batches
    total_code_files = len(categorized_files['code'])
    for i in range(0, total_code_files, batch_size):
        batch = categorized_files['code'][i:i + batch_size]
        logger.info(f"Processing code batch {i//batch_size + 1}/{(total_code_files + batch_size - 1)//batch_size}")
        log_memory_usage()
        
        batch_vectors = process_batch(batch, "code")
        vectors["code"].extend(batch_vectors)
        
        # Clear memory after each batch
        clear_memory()
        log_memory_usage()
        
        # Sleep between batches to let system resources recover
        logger.info(f"Sleeping for {sleep_time} seconds between batches...")
        time.sleep(sleep_time)
        
        logger.info(f"Progress: {min(i + batch_size, total_code_files)}/{total_code_files} code files processed")

    # Process doc files in batches
    total_doc_files = len(categorized_files['docs'])
    for i in range(0, total_doc_files, batch_size):
        batch = categorized_files['docs'][i:i + batch_size]
        logger.info(f"Processing docs batch {i//batch_size + 1}/{(total_doc_files + batch_size - 1)//batch_size}")
        log_memory_usage()
        
        batch_vectors = process_batch(batch, "docs")
        vectors["docs"].extend(batch_vectors)
        
        # Clear memory after each batch
        clear_memory()
        log_memory_usage()
        
        # Sleep between batches
        logger.info(f"Sleeping for {sleep_time} seconds between batches...")
        time.sleep(sleep_time)
        
        logger.info(f"Progress: {min(i + batch_size, total_doc_files)}/{total_doc_files} doc files processed")

    return vectors

def save_progress(vectors, output_file="vector_progress.pkl"):
    """Save progress to disk in case of interruption"""
    import pickle
    with open(output_file, 'wb') as f:
        pickle.dump(vectors, f)
    logger.info(f"Progress saved to {output_file}")

def load_progress(input_file="vector_progress.pkl"):
    """Load previously saved progress"""
    import pickle
    if os.path.exists(input_file):
        with open(input_file, 'rb') as f:
            vectors = pickle.load(f)
        logger.info(f"Loaded progress from {input_file}")
        return vectors
    return {"code": [], "docs": []}

# Helper function to check if the embedding generation is working
def test_embedding_generation(file_path):
    try:
        logger.info(f"Testing embedding generation for: {file_path}")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if file_path.endswith(('.py', '.js', '.java', '.cpp', '.c', '.h', '.go', '.rb')):
            embedding = get_graphcode_embedding(content)
        else:
            embedding = generate_text_embedding(content)
            
        if embedding is not None:
            logger.info("Test successful - embedding generated")
            return True
        else:
            logger.error("Test failed - embedding is None")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# def generate_vectors_for_repo(categorized_files):
#     vectors = {"code": [], "docs": []}

#     # Process code files
#     for code_file in categorized_files["code"]:
#         with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
#             code_content = f.read()
#             code_embedding = get_graphcode_embedding(code_content)
#             vectors["code"].append((code_file, code_embedding))

#     # Process document files
#     for doc_file in categorized_files["docs"]:
#         with open(doc_file, 'r', encoding='utf-8', errors='ignore') as f:
#             doc_content = f.read()
#             doc_embedding = generate_text_embedding(doc_file)
#             vectors["docs"].append((doc_file, doc_embedding))

#     return vectors

def process_embeddings(embeddings_data):
    processed_data = []
    
    # Process each category
    for category, files in embeddings_data.items():
        for file_name, embedding in files:
            # Convert embedding to list format
            if torch.is_tensor(embedding):
                # Handle PyTorch tensor (shape [1, 768])
                embedding = embedding.detach().cpu().numpy().flatten()
            elif isinstance(embedding, np.ndarray):
                # Handle numpy array directly
                embedding = embedding.flatten()

            # Convert to list format for consistency
            embedding = embedding.tolist()
            
            processed_data.append({
                'id': f"{category}_{file_name}",
                'embedding': embedding,
                'document': file_name,
                'metadata': {
                    'category': category,
                    'file_name': file_name
                }
            })
    
    return processed_data

# Separate processed data by embedding dimension
def add_data_chroma(processed_data):
    code_data = [item for item in processed_data if len(item['embedding']) == 768]
    docs_data = [item for item in processed_data if len(item['embedding']) == 384]

    # Add code embeddings (768 dimension)
    code_collection.add(
        embeddings=[item['embedding'] for item in code_data],
        documents=[item['document'] for item in code_data],
        ids=[item['id'] for item in code_data],
        metadatas=[item['metadata'] for item in code_data]
    )
    docs_collection.add(
        embeddings=[item['embedding'] for item in docs_data],
        documents=[item['document'] for item in docs_data],
        ids=[item['id'] for item in docs_data],
        metadatas=[item['metadata'] for item in docs_data]
    )

    print("data successfully add to chroma db")

def query_chroma(code_query_embedding):
    # Set the number of nearest neighbors to retrieve (k)
    k = 3

    code_results = code_collection.query(
        query_embeddings=[code_query_embedding.tolist()],  
        n_results=k,
    )

    # Check results
    return(code_results)