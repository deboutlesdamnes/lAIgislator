from llama_index import Document, LLMPredictor, LangchainEmbedding, SimpleDirectoryReader, VectorStoreIndex, ServiceContext, StorageContext, load_graph_from_storage, set_global_service_context
from llama_index.llms import HuggingFaceLLM, LangChainLLM
from llama_index.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.query_engine.multistep_query_engine import MultiStepQueryEngine
from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp, HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from llama_index.vector_stores import ChromaVectorStore
from chromadb.config import Settings
from llama_index.indices.composability import ComposableGraph
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline, logging
import logging
import sys
import chromadb

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

n_gpu_layers = 100  # Change this value based on your model and your GPU VRAM pool.
n_batch = 256

tokenizer = AutoTokenizer.from_pretrained("upstage/Llama-2-70b-instruct-v2")
model = AutoModelForCausalLM.from_pretrained(
    "upstage/Llama-2-70b-instruct-v2",
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    rope_scaling={"type": "dynamic", "factor": 2} # allows handling of longer inputs
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=pipe)


'''llm = LangChainLLM(llm=LlamaCpp(
    model_path="/home/jason/llama.cpp/models/llama-2-13b-chat.ggmlv3.q4_0.bin",
    input={"temperature": 0.75, "max_length": 512, "top_p": 1},
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    verbose=True,
    n_ctx=4096
))
'''

chroma_client = chromadb.PersistentClient(path="/home/pebble/lai/bill_index", settings=Settings(
    anonymized_telemetry=False
))

'''
chroma_collection = chroma_client.get_collection("billtexts_full")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="/home/pebble/lai/bill_index")
'''

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)


index = ComposableGraph.load_from_disk("/home/pebble/lai/graph.json")

query_engine = index.as_query_engine(service_context=service_context)

response = query_engine.query('Find me a bill that seeks to reduce the number of school shootings')
