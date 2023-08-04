from llama_index import Document, LangchainEmbedding, SimpleDirectoryReader, VectorStoreIndex, ServiceContext, set_global_service_context
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.node_parser import SimpleNodeParser
import glob
import os

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)
service_context = ServiceContext.from_defaults(embed_model=embed_model)
set_global_service_context(service_context)

print("defining things...")
documents = SimpleDirectoryReader('./billtext').load_data()
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)
index.storage_context.persist(persist_dir="<persist_dir>")

print("loading documents...")
text_list = []
text_path = r'./billtext'
for file in os.listdir(text_path):
    if file.endswith('.txt'):
        text_list.append(file)

documents = [Document(text=t) for t in text_list]



