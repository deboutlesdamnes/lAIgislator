from llama_index import Document, LLMPredictor, LangchainEmbedding, SimpleDirectoryReader, VectorStoreIndex, ServiceContext, StorageContext, SimpleKeywordTableIndex, load_index_from_storage
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
from langchain import OpenAI
from llama_index import Document, LangchainEmbedding, SimpleDirectoryReader, SimpleKeywordTableIndex, StorageContext, VectorStoreIndex, GPTVectorStoreIndex, ServiceContext, set_global_service_context, load_index_from_storage
from llama_index.llms import HuggingFaceLLM, LangChainLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp, HuggingFacePipeline
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.vector_stores import ChromaVectorStore
from llama_index.graph_stores import Neo4jGraphStore
from IPython.display import Markdown, display
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.node_parser import SimpleNodeParser
from llama_index.indices.composability import ComposableGraph
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline
import logging
import sys
import chromadb
import torch
import os
from bs4 import BeautifulSoup
from process_xml import get_title, get_billtext, get_chamber, get_status, get_date
from chromadb.config import Settings

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

model_name_or_path = "TheBloke/Llama-2-70B-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

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

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, chunk_size=1024)
set_global_service_context(service_context)

con_hr1131 = StorageContext.from_defaults(persist_dir = "/home/pebble/lai/bill_indexes/hr1131")
con_hr1132 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/hr1132')
con_s1131 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/s1131')
con_s1132 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/s1132')
con_hr1141 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/hr1141')
con_hr1142 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/hr1142')
con_s1141 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/s1141')
con_s1142 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/s1142')
con_hr1151 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/hr1151')
con_hr1152 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/hr1152')
con_s1151 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/s1151')
con_s1152 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/s1152')
con_hr1161 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/hr1161')
con_hr1162 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/hr1162')
con_s1161 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/s1161')
con_s1162 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/s1162')
con_hr1171 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/hr1171')
con_hr1172 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/hr1172')
con_s1171 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/s1171')
con_s1172 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/s1172')
con_hr1181 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/hr1181')
con_s1181 = StorageContext.from_defaults(persist_dir = '/home/pebble/lai/bill_indexes/s1181')

hr1131 = load_index_from_storage(con_hr1131) 
hr1132 = load_index_from_storage(con_hr1132) 
hr1141 = load_index_from_storage(con_hr1141) 
hr1142 = load_index_from_storage(con_hr1142) 
hr1151 = load_index_from_storage(con_hr1151) 
hr1152 = load_index_from_storage(con_hr1152) 
hr1161 = load_index_from_storage(con_hr1161) 
hr1162 = load_index_from_storage(con_hr1162) 
hr1171 = load_index_from_storage(con_hr1171) 
hr1172 = load_index_from_storage(con_hr1172) 
hr1181 = load_index_from_storage(con_hr1181) 
s1131 = load_index_from_storage(con_s1131) 
s1132 = load_index_from_storage(con_s1132) 
s1141 = load_index_from_storage(con_s1141) 
s1142 = load_index_from_storage(con_s1142) 
s1151 = load_index_from_storage(con_s1151) 
s1152 = load_index_from_storage(con_s1152) 
s1161 = load_index_from_storage(con_s1161) 
s1162 = load_index_from_storage(con_s1162) 
s1171 = load_index_from_storage(con_s1171) 
s1172 = load_index_from_storage(con_s1172) 
s1181 = load_index_from_storage(con_s1181) 


hr1131_summary = """House Resolutions from the first session of the 113th Congress, which lasted from 2013/01/03 to 2014/01/03 and saw Republicans control the House"""
hr1132_summary = """House Resolutions from the second session of the 113th Congress, which lasted from 2014/01/03 to 2015/01/03 and saw Republicans control the House"""
s1131_summary = """Senate bills from the first session of the 113th Congress, which lasted from 2013/01/03 to 2014/01/03 and saw Democrats control the Senate"""
s1132_summary = """Senate bills from the second session of the 113th Congress, which lasted from 2014/01/03 to 2015/01/03 and saw Democrats control the Senate"""
hr1141_summary = """House Resolutions from the first session of the 114th Congress, which lasted from 2015/01/03 to 2016/01/03 and saw Republicans control the House"""
hr1142_summary = """House Resolutions from the second session of the 114th Congress, which lasted from 2016/01/03 to 2017/01/03 and saw Republicans control the House"""
s1141_summary = """Senate bills from the first session of the 114th Congress, which lasted from 2015/01/03 to 2016/01/03 and saw Republicans control the Senate"""
s1142_summary = """Senate bills from the second session of the 114th Congress, which lasted from 2016/01/03 to 2017/01/03 and saw Republicans control the Senate"""
hr1151_summary = """House Resolutions from the first session of the 115th Congress, which lasted from 2017/01/03 to 2018/01/03 and saw Republicans control the House"""
hr1152_summary = """House Resolutions from the second session of the 115th Congress, which lasted from 2018/01/03 to 2019/01/03 and saw Republicans control the House"""
s1151_summary = """Senate bills from the first session of the 115th Congress, which lasted from 2017/01/03 to 2018/01/03 and saw Republicans control the Senate"""
s1152_summary = """Senate bills from the second session of the 115th Congress, which lasted from 2018/01/03 to 2019/01/03 and saw Republicans control the Senate"""
hr1161_summary = """House Resolutions from the first session of the 116th Congress, which lasted from 2019/01/03 to 2020/01/03 and saw Democrats control the House"""
hr1162_summary = """House Resolutions from the second session of the 116th Congress, which lasted from 2020/01/03 to 2021/01/03 and saw Democrats control the House"""
s1161_summary = """Senate bills from the first session of the 116th Congress, which lasted from 2019/01/03 to 2020/01/03 and saw Republicans control the Senate"""
s1162_summary = """Senate bills from the second session of the 116th Congress, which lasted from 2020/01/03 to 2021/01/03 and saw Republicans control the Senate"""
hr1171_summary = """House Resolutions from the first session of the 117th Congress, which lasted from 2021/01/03 to 2022/01/03 and saw Democrats control the House"""
hr1172_summary = """House Resolutions from the second session of the 117th Congress, which lasted from 2022/01/03 to 2023/01/03 and saw Democrats control the House"""
s1171_summary = """Senate bills from the first session of the 117th Congress, which lasted from 2021/01/03 to 2022/01/03 and saw Republicans control the Senate until 2021/01/20 and Democrats control the Senate for the rest of the session"""
s1172_summary = """Senate bills from the second session of the 117th Congress, which lasted from 2022/01/03 to 2023/01/03 and saw Republicans control the Senate"""
hr1181_summary = """House Resolutions from the first session of the 118th Congress, which lasted from 2023/01/03 to 2024/01/03 and saw Republicans control the House"""
s1181_summary = """Senate bills from the first session of the 118th Congress, which lasted from 2023/01/03 to 2024/01/03 and saw Democrats control the Senate"""

graph = ComposableGraph.from_indices(
    SimpleKeywordTableIndex,
    [hr1131, s1131, hr1132, s1132, hr1141, s1141, hr1142, s1142, hr1151, s1151, hr1152, s1152, hr1161, s1161, hr1162, s1162, hr1171, s1171, hr1172, s1172, hr1181, s1181],
    index_summaries=[hr1131_summary, s1131_summary, hr1132_summary, s1132_summary, hr1141_summary, s1141_summary, hr1142_summary, s1142_summary, hr1151_summary, s1151_summary, hr1152_summary, s1152_summary, hr1161_summary, s1161_summary, hr1162_summary, s1162_summary, hr1171_summary, s1171_summary, hr1172_summary, s1172_summary, hr1181_summary, s1181_summary],
    max_keywords_per_chunk=50,
)

'''
chroma_collection = chroma_client.get_collection("billtexts_full")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="/home/pebble/lai/bill_index")
'''

query_engine = graph.as_query_engine(service_context=service_context)

response = query_engine.query('Find me a bill that seeks to reduce the number of school shootings')
