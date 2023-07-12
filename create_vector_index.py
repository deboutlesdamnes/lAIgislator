from typing import Dict, List
from pathlib import Path
import langchain
import llama_index
from llama_index import download_loader
from llama_index import Document

# Step 1: Logic for loading and parsing the files into llama_index documents.
UnstructuredReader = download_loader("UnstructuredReader")
loader = UnstructuredReader()

print("Loaded readers...")
# Step 2: Convert the loaded documents into llama_index Nodes. This will split the documents into chunks.
from llama_index.node_parser import SimpleNodeParser
from llama_index.data_structs import Node

def convert_documents_into_nodes(documents: Dict[str, Document]) -> Dict[str, Node]:
    parser = SimpleNodeParser()
    document = documents["doc"]
    nodes = parser.get_nodes_from_documents([document])
    return [{"node": node} for node in nodes]


# Step 3: Embed each node using a local embedding model.
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

class EmbedNodes:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            # Use all-mpnet-base-v2 Sentence_transformer.
            # This is the default embedding model for LlamaIndex/Langchain.
            model_name="sentence-transformers/all-mpnet-base-v2", 
            model_kwargs={"device": "cuda"},
            # Use GPU for embedding and specify a large enough batch size to maximize GPU utilization.
            # Remove the "device": "cuda" to use CPU instead.
            encode_kwargs={"device": "cuda", "batch_size": 100}
            )
    
    def __call__(self, node_batch: Dict[str, List[Node]]) -> Dict[str, List[Node]]:
        nodes = node_batch["node"]
        text = [node.text for node in nodes]
        embeddings = self.embedding_model.embed_documents(text)
        assert len(nodes) == len(embeddings)

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
        return {"embedded_nodes": nodes}

# Step 4: Stitch together all of the above into a Ray Data pipeline.
import ray
from ray.data import ActorPoolStrategy

# First, download the Ray documentation locally
ds = ray.data.read_text("s3://docs-public-bills/")
print("Documents loaded...")
# Use `flat_map` since there is a 1:N relationship. Each document returns multiple nodes.
nodes = ds.map_batches(convert_documents_into_nodes)
print("Converted to nodes...")
# Use `map_batches` to specify a batch size to maximize GPU utilization.
# We define `EmbedNodes` as a class instead of a function so we only initialize the embedding model once. 
#   This state can be reused for multiple batches.
embedded_nodes = nodes.map_batches(
    EmbedNodes, 
    batch_size=100, 
    # Use 1 GPU per actor.
    num_gpus=1,
    # There are 4 GPUs in the cluster. Each actor uses 1 GPU. So we want 4 total actors.
    # Set the size of the ActorPool to the number of GPUs in the cluster.
    compute=ActorPoolStrategy(size=4),
    )

# Step 5: Trigger execution and collect all the embedded nodes.
h117_nodes = []
for row in embedded_nodes.iter_rows():
    node = row["embedded_nodes"]
    assert node.embedding is not None
    h117_nodes.append(node)
print("Nodes embedded...")

# Step 6: Store the embedded nodes in a local vector store, and persist to disk.
print("Storing in vector index...")
from llama_index import chromadb
ray_docs_index = chromadb(nodes=h117_nodes)
ray_docs_index.storage_context.persist(persist_dir="/tmp/ray_docs_index")
