import chromadb

client = chromadb.PersistentClient(path="db/")

code_collection = client.get_or_create_collection("code_embeddings_768")
docs_collection = client.get_or_create_collection("docs_embeddings_384")
