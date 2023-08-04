from llama_index import Document, LangchainEmbedding, SimpleDirectoryReader, VectorStoreIndex, ServiceContext, set_global_service_context
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.node_parser import SimpleNodeParser
import torch
import glob
import os

query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

llm = HuggingFaceLLM(
    context_window=4096, 
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="TheBloke/Llama-2-70B-GGML",
    model_name="TheBloke/Llama-2-70B-GGML",
    device_map="auto",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
    model_kwargs={"torch_dtype": torch.float16}
)

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



