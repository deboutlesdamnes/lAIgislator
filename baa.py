from langchain import OpenAI
from llama_index import Document, LangchainEmbedding, SimpleDirectoryReader, SimpleKeywordTableIndex, StorageContext, VectorStoreIndex, GPTVectorStoreIndex, ServiceContext, set_global_service_context
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

'''
username = "neo4j"
password = "retractor-knot-thermocouples"
url = "bolt://44.211.44.239:7687"
database = "neo4j"
'''

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

logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

'''

tokenizer = AutoTokenizer.from_pretrained("upstage/Llama-2-70b-instruct-v2")
model = AutoModelForCausalLM.from_pretrained(
    "upstage/Llama-2-70b-instruct-v2",
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    rope_scaling={"type": "dynamic", "factor": 2} # allows handling of longer inputs
)

pipe = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)
'''

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


#llm = OpenAI(temperature=0, model="text-davinci-002", max_tokens=512, openai_api_key="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")

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, chunk_size=1024)
set_global_service_context(service_context)

'''
chroma_client = chromadb.PersistentClient(path="/home/pebble/lai/bill_index")
chroma_collection = chroma_client.get_or_create_collection("billtexts_full")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

graph_store = Neo4jGraphStore(
    username=username,
    password=password,
    url=url,
    database=database,
)
'''


storage_context = StorageContext.from_defaults()

print("loading documents...")

def create_index(doc_path):
    doc_list = []
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
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context, show_progress=True)
    file_base = os.path.basename(doc_path)
    file_name = os.path.splitext(file_base)[0]
    storage_context.persist(persist_dir="/home/pebble/lai/bill_indexes/" + file_name)

# Creating indexes for each session and each chamber

hr1131 = create_index('/home/pebble/lai/hr1131')
hr1132 = create_index('/home/pebble/lai/hr1132')
s1131 = create_index('/home/pebble/lai/s1131')
s1132 = create_index('/home/pebble/lai/s1132')
hr1141 = create_index('/home/pebble/lai/hr1141')
hr1142 = create_index('/home/pebble/lai/hr1142')
s1141 = create_index('/home/pebble/lai/s1141')
s1142 = create_index('/home/pebble/lai/s1142')
hr1151 = create_index('/home/pebble/lai/hr1151')
hr1152 = create_index('/home/pebble/lai/hr1152')
s1151 = create_index('/home/pebble/lai/s1151')
s1152 = create_index('/home/pebble/lai/s1152')
hr1161 = create_index('/home/pebble/lai/hr1161')
hr1162 = create_index('/home/pebble/lai/hr1162')
s1161 = create_index('/home/pebble/lai/s1161')
s1162 = create_index('/home/pebble/lai/s1162')
hr1171 = create_index('/home/pebble/lai/hr1171')
hr1172 = create_index('/home/pebble/lai/hr1172')
s1171 = create_index('/home/pebble/lai/s1171')
s1172 = create_index('/home/pebble/lai/s1172')
hr1181 = create_index('/home/pebble/lai/hr1181')
s1181 = create_index('/home/pebble/lai/s1181')

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

'''

graph = ComposableGraph.from_indices(
    SimpleKeywordTableIndex,
    [hr1131, s1131, hr1132, s1132, hr1141, s1141, hr1142, s1142, hr1151, s1151, hr1152, s1152, hr1161, s1161, hr1162, s1162, hr1171, s1171, hr1172, s1172, hr1181, s1181],
    index_summaries=[hr1131_summary, s1131_summary, hr1132_summary, s1132_summary, hr1141_summary, s1141_summary, hr1142_summary, s1142_summary, hr1151_summary, s1151_summary, hr1152_summary, s1152_summary, hr1161_summary, s1161_summary, hr1162_summary, s1162_summary, hr1171_summary, s1171_summary, hr1172_summary, s1172_summary, hr1181_summary, s1181_summary],
    max_keywords_per_chunk=50,
)

graph.root_index.set_index_id("bill_graph_root")

# persist to storage
graph.root_index.storage_context.persist(persist_dir="./storage")
'''










