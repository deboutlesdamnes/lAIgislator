from llama_index import Document, SimpleDirectoryReader, VectorStoreIndex
from llama_index.node_parser import SimpleNodeParser
import glob
import os

documents = SimpleDirectoryReader('./billtexts').load_data()
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)
index.storage_context.persist(persist_dir="<persist_dir>")

text_list = []
text_path = r'./billtexts'
for file in os.listdir(text_path):
    if file.endswith('.txt'):
        text_list.append(file)

documents = [Document(text=t) for t in text_list]

#testing commit

