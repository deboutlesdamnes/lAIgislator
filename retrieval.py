import json
import chromadb
from chromadb.utils import embedding_functions
from llama_index.vector_stores import ChromaVectorStore

client = chromadb.PersistentClient(path="/home/jason/lai/bill_index")
collection = client.get_collection(name="billtexts_full", embedding_function=embedding_functions.DefaultEmbeddingFunction())
docs = collection.get(include=['metadatas'])
with open ('output.json', 'w') as output:
    json.dump(docs, output)
