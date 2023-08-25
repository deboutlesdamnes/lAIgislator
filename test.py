import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index import VectorStoreIndex, SimpleKeywordTableIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores import ChromaVectorStore
# fetch "New York City" page from Wikipedia
from pathlib import Path
import chromadb

import requests

response = requests.get(
    "https://en.wikipedia.org/w/api.php",
    params={
        "action": "query",
        "format": "json",
        "titles": "New York City",
        "prop": "extracts",
        # 'exintro': True,
        "explaintext": True,
    },
).json()
page = next(iter(response["query"]["pages"].values()))
nyc_text = page["extract"]

data_path = Path("data")
if not data_path.exists():
    Path.mkdir(data_path)

with open("../test_wiki/data/nyc_text.txt", "w") as fp:
    fp.write(nyc_text)
# load NYC dataset
nyc_documents = SimpleDirectoryReader("../test_wiki/data/").load_data()
# load PG's essay
essay_documents = SimpleDirectoryReader("../paul_graham_essay/data/").load_data()


# build NYC index
nyc_index = VectorStoreIndex.from_documents(nyc_documents)

# build essay index
essay_index = VectorStoreIndex.from_documents(essay_documents)


nyc_index_summary = """
    New York, often called New York City or NYC, 
    is the most populous city in the United States. 
    With a 2020 population of 8,804,190 distributed over 300.46 square miles (778.2 km2), 
    New York City is also the most densely populated major city in the United States, 
    and is more than twice as populous as second-place Los Angeles. 
    New York City lies at the southern tip of New York State, and 
    constitutes the geographical and demographic center of both the 
    Northeast megalopolis and the New York metropolitan area, the 
    largest metropolitan area in the world by urban landmass.[8] With over 
    20.1 million people in its metropolitan statistical area and 23.5 million 
    in its combined statistical area as of 2020, New York is one of the world's 
    most populous megacities, and over 58 million people live within 250 mi (400 km) of 
    the city. New York City is a global cultural, financial, and media center with 
    a significant influence on commerce, health care and life sciences, entertainment, 
    research, technology, education, politics, tourism, dining, art, fashion, and sports. 
    Home to the headquarters of the United Nations, 
    New York is an important center for international diplomacy,
    an established safe haven for global investors, and is sometimes described as the capital of the world.
"""
essay_index_summary = """
    Author: Paul Graham. 
    The author grew up painting and writing essays. 
    He wrote a book on Lisp and did freelance Lisp hacking work to support himself. 
    He also became the de facto studio assistant for Idelle Weber, an early photorealist painter. 
    He eventually had the idea to start a company to put art galleries online, but the idea was unsuccessful. 
    He then had the idea to write software to build online stores, which became the basis for his successful company, Viaweb. 
    After Viaweb was acquired by Yahoo!, the author returned to painting and started writing essays online. 
    He wrote a book of essays, Hackers & Painters, and worked on spam filters. 
    He also bought a building in Cambridge to use as an office. 
    He then had the idea to start Y Combinator, an investment firm that would 
    make a larger number of smaller investments and help founders remain as CEO. 
    He and his partner Jessica Livingston ran Y Combinator and funded a batch of startups twice a year. 
    He also continued to write essays, cook for groups of friends, and explore the concept of invented vs discovered in software. 
"""

from llama_index.indices.composability import ComposableGraph
graph = ComposableGraph.from_indices(
    SimpleKeywordTableIndex,
    [nyc_index, essay_index],
    index_summaries=[nyc_index_summary, essay_index_summary],
    max_keywords_per_chunk=50,
)

chroma_client = chromadb.PersistentClient(path="/home/pebble/lai/bill_index")
chroma_collection = chroma_client.get_or_create_collection("billtexts_full")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

graph.storage_context.persist(persist_dir="/home/pebble/lai/bill_index")