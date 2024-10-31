from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer
from unixcoder import UniXcoder

import torch

# # Load GraphCodeBERT model and tokenizer
# tokenizer = UniXcoder("microsoft/unixcoder-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the UniXcoder model
model = UniXcoder("microsoft/unixcoder-base")
model.to(device)

# without uniXcoder
# tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
# model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")

def get_graphcode_embedding(code_snippet):
    tokens_ids = model.tokenize([code_snippet],max_length=512,mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    tokens_embeddings,max_func_embedding = model(source_ids)
    return max_func_embedding

def query_emebdding(code_snippet):
    tokens_ids = model.tokenize([code_snippet], max_length=512, mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    
    # Perform forward pass through the model
    with torch.no_grad():
        tokens_embeddings, max_func_embedding = model(source_ids)
    
    # Convert the max_func_embedding to a NumPy array
    max_func_embedding_np = max_func_embedding.detach().cpu().numpy()
    
    return max_func_embedding_np.squeeze()

# # Function to generate embeddings
# def get_graphcode_embedding(code_snippet):
#     tokens_ids = model.tokenize([code_snippet], max_length=512, mode="<decoder-only>")
#     outputs = model(tokens_ids)
#     embedding = outputs[0].mean(dim=1).detach()  # Mean pooling to get the embedding
#     return embedding

text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to generate embeddings for text files
def generate_text_embedding(text_snippet):
    return text_model.encode(text_snippet)