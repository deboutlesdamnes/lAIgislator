import io
import pypdf
import ray
import langchain
from langchain.vectorstores import chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from pypdf import PdfReader

ray.init(
    runtime_env={"pip": ["langchain", "pypdf", "sentence_transformers", "transformers"]}
)

#texts from the 117th Congress
docs = ray.data.read_text("s3://ray-llm-batch-inference/")

#splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)
texts = text_splitter.split_documents(docs)

from sentence_transformers import SentenceTransformer

# Use LangChain's default model.
# This model can be changed depending on your task.
model_name = "sentence-transformers/all-mpnet-base-v2"


# We use sentence_transformers directly to provide a specific batch size.
# LangChain's HuggingfaceEmbeddings can be used instead once https://github.com/hwchase17/langchain/pull/3914
# is merged.
class Embed:
    def __init__(self):
        # Specify "cuda" to move the model to GPU.
        self.transformer = SentenceTransformer(model_name, device="cuda")

    def __call__(self, text_batch: List[str]):
        # We manually encode using sentence_transformer since LangChain
        # HuggingfaceEmbeddings does not support specifying a batch size yet.
        embeddings = self.transformer.encode(
            text_batch,
            batch_size=100,  # Large batch size to maximize GPU utilization.
            device="cuda",
        ).tolist()

        return list(zip(text_batch, embeddings))


# Use `map_batches` since we want to specify a batch size to maximize GPU utilization.
ds = ds.map_batches(
    Embed,
    # Large batch size to maximize GPU utilization.
    # Too large a batch size may result in GPU running out of memory.
    # If the chunk size is increased, then decrease batch size.
    # If the chunk size is decreased, then increase batch size.
    batch_size=50,  # Large batch size to maximize GPU utilization.
    compute=ray.data.ActorPoolStrategy(min_size=3, max_size=3),  # I have 20 GPUs in my cluster
    num_gpus=1,  # 1 GPU for each actor.
)

persist_directory = '/root/chromaDB/lAIgislator'
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
print("Creating the Database...")

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)

print("Saving...")
# persist the db to disk
vectordb.persist()
vectordb = None
