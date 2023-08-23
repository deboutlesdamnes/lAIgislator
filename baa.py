from langchain import OpenAI
from llama_index import Document, LangchainEmbedding, SimpleDirectoryReader, StorageContext, VectorStoreIndex, ServiceContext, set_global_service_context
from llama_index.llms import HuggingFaceLLM, LangChainLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.vector_stores import ChromaVectorStore
from llama_index.vector_stores.faiss import FaissVectorStore
from IPython.display import Markdown, display
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.node_parser import SimpleNodeParser
import chromadb
import openai
import glob
import os
import logging
import sys
import faiss
from bs4 import BeautifulSoup
from process_xml import get_title, get_billtext, get_chamber, get_status, get_date
from chromadb.config import Settings

#query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
#callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#n_gpu_layers = 100  # Change this value based on your model and your GPU VRAM pool.
#n_batch = 1024

"""Local llamaCpp:
llm = LangChainLLM(llm=LlamaCpp(
    model_path="/home/jason/llama.cpp/models/llama-2-13b-chat.ggmlv3.q4_0.bin",
    input={"temperature": 0.75, "max_length": 512, "top_p": 1},
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,
    n_ctx=1024
))
"""

"""HuggingFace LLM:
llm = HuggingFaceLLM(
    context_window=4096, 
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
    model_name="StabilityAI/stablelm-tuned-alpha-3b",
    device_map="auto",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
    # uncomment this if using CUDA to reduce memory usage
    # model_kwargs={"torch_dtype": torch.float16}
)
"""

#llm = OpenAI(temperature=0, model="text-davinci-002", max_tokens=512, openai_api_key="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")

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

service_context = ServiceContext.from_defaults(embed_model=embed_model, chunk_size=1024)
set_global_service_context(service_context)

chroma_client = chromadb.PersistentClient(path="/home/jason/lai/bill_index")
chroma_collection = chroma_client.get_or_create_collection("billtexts_full")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("loading documents...")
doc_list = []
doc_path = '/home/jason/lai/bills-archive'
documents = []
for file in os.listdir(doc_path):
    if file.endswith('.xml'):
        doc_list.append(file)
for doc in doc_list: 
    with open (os.path.join(doc_path, doc), 'r', encoding='utf-8') as input:
        soup = BeautifulSoup(input, features="xml")
        loaded_doc = Document(
            text = get_billtext(soup),
            metadata={
                'title': get_title(soup), 
                'status': get_status(soup),
                'passed date': get_date(soup),
                'chamber': get_chamber(soup),
            })
        documents.append(loaded_doc)

'''
filedata_fn = lambda filename: {'file_name': filename,
                                'chamber': 'House'}
print("loading data")
documents = SimpleDirectoryReader('/home/jason/lai/billtext', file_metadata=filedata_fn, filename_as_id=True).load_data()
'''

#parser = SimpleNodeParser()
#print("parsing data")
#nodes = parser.get_nodes_from_documents(documents, show_progress=True)
print("creating vector index")
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context, show_progress=True)








