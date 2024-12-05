#server configurations
import streamlit as st
import subprocess
import redis
import uuid
from typing import Optional, List
from llama_index.readers.web import SimpleWebPageReader, BeautifulSoupWebReader, RssReader, WholeSiteReader
from llama_index.core import SummaryIndex
import json
from llama_index.readers.file import (
    DocxReader,
    PDFReader,
    FlatReader,
    HTMLTagReader,
    ImageReader,
    IPYNBReader,
    PptxReader,
    PandasCSVReader,
    PyMuPDFReader,
    XMLReader,
    CSVReader,
)
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.vector_stores.milvus.utils import get_default_sparse_embedding_function
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
import os
from pydantic import BaseModel
from typing import Dict, Any
from pymilvus import connections
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore,IndexManagement 
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore
from typing import Optional
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List
from llama_index.llms.ollama import Ollama

import uuid
import redis
import json

# Connect to Redis
redis_client =""
task_data_name=""

class Task(BaseModel):
    description: str
    status: str

st.title("RAG Chat Application")
st.sidebar.header("Configurations")

# Function to get installed Ollama models using the 'ollama list' command
def get_ollama_models():
    try:
        # Execute the 'ollama list' command and capture the output
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        
        # Parse the output (assuming the models are listed line by line)
        models = result.stdout.splitlines()
        
        # Filter out empty lines or unnecessary information (if needed)
        model_names = []
        for model in models:
            model = model.strip()
            if model:  # Skip empty lines
                # Split by the first space and take the part before the first space
                model_name = model.split()[0]
                model_names.append(model_name)

        return model_names
    except subprocess.CalledProcessError as e:
        st.error(f"Error occurred while trying to fetch models: {e}")
        return []

# Get the available models using 'ollama list'
available_models = get_ollama_models()

parameters={}

if available_models:
    # Dropdown to select the model
    parameters["selected_model"] = st.selectbox("Select an Ollama Model", available_models)

# Task Management Section
st.sidebar.subheader("Task Management Configuration")
parameters["task_data_management_collection"] = st.sidebar.text_input("Task Management Collection Name", value="task_data")
parameters["taskHost"] = st.sidebar.text_input("Task Management Host", value="localhost")
parameters["taskPort"] = st.sidebar.number_input("Task Management Port", value=6379)

# Embedding Model Section
st.sidebar.subheader("Embedding Model Configuration")
parameters["modelName"] = st.sidebar.text_input("Model Name", value="BAAI/bge-small-en-v1.5")
parameters["modelBaseUrl"] = st.sidebar.text_input("Base URL", value="http://127.0.0.1:8081")
parameters["modelEmbedBatchSize"] = st.sidebar.number_input("Batch Size", value=10)
parameters["modelTimeout"] = st.sidebar.number_input("Timeout (s)", step=1.0, format="%.2f", value=60.0)
parameters["modelTruncateText"] = st.sidebar.checkbox("Truncate Text", value=True)

# Document Store Section
st.sidebar.subheader("Document Store Configuration")
parameters["docHost"] = st.sidebar.text_input("Doc Host", value="localhost")
parameters["docPort"] = st.sidebar.number_input("Doc Port", value=6379)
parameters["docCollectionName"] = st.sidebar.text_input("Doc Collection Name", value="document_store")

# Cache Configurations Section
st.sidebar.subheader("Cache Configuration")
parameters["cacheHost"] = st.sidebar.text_input("Cache Host", value="127.0.0.1")
parameters["cachePort"] = st.sidebar.number_input("Cache Port", value=6379)
parameters["cacheCollectionName"] = st.sidebar.text_input("Cache Collection Name", value="cache")

# Vector Store Configurations Section
st.sidebar.subheader("Vector Store Configuration")
parameters["milvusURI"] = st.sidebar.text_input("Milvus URI", value="http://localhost:19530")
parameters["milvusDimension"] = st.sidebar.number_input("Dimension", value=1024)
parameters["milvusOverwrite"] = st.sidebar.checkbox("Overwrite", value=False)
parameters["milvusCollectionName"] = st.sidebar.text_input("Collection Name", value="app_milvus_db")
parameters["milvusConsistencyLevel"] = st.sidebar.selectbox("Consistency Level", ["Eventually", "Strong", "Bounded", "Session"])
parameters["milvusEnableSparse"] = st.sidebar.checkbox("Enable Sparse", value=False)
parameters["milvusSimilarityMetric"] = st.sidebar.selectbox("Similarity Metric", ["IP", "L2"])
parameters["milvusEmbeddingField"] = st.sidebar.text_input("Embedding Field", value="embedding")
parameters["milvusDocIdField"] = st.sidebar.text_input("Doc ID Field", value="doc_id")
parameters["milvusTextKey"] = st.sidebar.text_input("Text Key", value="text_key")
parameters["milvusSparseEmbeddingField"] = st.sidebar.text_input("Sparse Embedding Field", value="sparse_embedding")
parameters["milvusBatchSize"] = st.sidebar.number_input("Batch Size", value=100)

# Content Ingestion Section
st.sidebar.subheader("Content Ingestion")
parameters["format"] = st.sidebar.selectbox("Format", ["web", "whole_site", "pdf", "docx", "txt", "image", "ipynb", "pptx", "csv", "xml", "rss", "beautiful_soup", "html_tags"])
if parameters["format"] == "web":
    parameters["url_web"] = st.sidebar.text_input("URL", value=None)
elif parameters["format"] == "html_tags":
    parameters["url_html"] = st.sidebar.text_input("HTML URL", value=None)
    parameters["tag"] = st.sidebar.text_input("Tag", value="section")
    parameters["ignore_no_id"] = st.sidebar.checkbox("Ignore No ID", value=True)
elif parameters["format"] == "beautiful_soup":
    parameters["url_soup"] = st.sidebar.text_input("Soup URL", value=None)
elif parameters["format"] == "whole_site":
    parameters["url_whole"] = st.sidebar.text_input("Base URL", value=None)
    parameters["prefix"] = st.sidebar.text_input("Prefix", value=None)
    parameters["max_depth"] = st.sidebar.number_input("Max Depth", value=10)
elif parameters["format"] in ["pdf", "docx", "txt", "image", "ipynb", "pptx", "csv", "xml"]:
    parameters["url_file"] = st.sidebar.text_input("File Path", value=None)

def create_task(description: str) -> str:
    redis_client = redis.Redis(host=parameters["taskHost"], port=parameters["taskPort"], db=0)
    task_id = str(uuid.uuid4())
    task_data = {"description": description, "status": "in process"}
    redis_client.hset(task_data_name, task_id, json.dumps(task_data))
    return task_id

def update_task_status(task_id: str, status: str):
    redis_client = redis.Redis(host=parameters["taskHost"], port=parameters["taskPort"], db=0)
    if redis_client.hexists(task_data_name, task_id):
        task_data = json.loads(redis_client.hget(task_data_name, task_id))
        task_data["status"] = status
        redis_client.hset(task_data_name, task_id, json.dumps(task_data))

def get_all_tasks():
    task_data_name=parameters["task_data_management_collection"]
    print(task_data_name)
    redis_client = redis.Redis(host=parameters["taskHost"], port=parameters["taskPort"], db=0)
    print(redis_client)
    task_data = redis_client.hgetall(task_data_name)
    tasks = {task_id.decode('utf-8'): json.loads(task_info) for task_id, task_info in task_data.items()}
    return tasks

def get_task(task_id: str):
    task_data_name=parameters["task_data_management_collection"]
    redis_client = redis.Redis(host=parameters["taskHost"], port=parameters["taskPort"], db=0)
    if redis_client.hexists(task_data_name, task_id):
        task_info = json.loads(redis_client.hget(task_data_name, task_id))
        return task_info
    else:
        raise Exception(status_code=404, detail="Task not found")

def read_data():
    global task_data_name
    global redis_client
    documents=None
    
    if parameters["format"] == "web":
        if parameters["url_web"] is None:
            raise Exception(status_code=400, detail="URL must be provided for web reader")
        documents = SimpleWebPageReader(html_to_text=True).load_data([parameters["url_web"]])
    
    elif parameters["format"] == "html_tags":
        if parameters["url_html"] is None:
            raise Exception(status_code=400, detail="URL must be provided for HTML tag reader")
        if not os.path.exists(parameters["url_html"]):
            raise Exception(status_code=400, detail="File not found")
        with open(parameters["url_html"], "r", encoding="utf-8") as f:
            content = f.read()
        reader = HTMLTagReader(tag=parameters["tag"], ignore_no_id=parameters["ignore_no_id"])
        documents = reader.load_data(content)
    
    elif parameters["format"] == "beautiful_soup":
        if parameters["url_soup"] is None:
            raise Exception(status_code=400, detail="URL must be provided for BeautifulSoup reader")
        loader = BeautifulSoupWebReader()
        documents = loader.load_data(urls=[parameters["url_soup"]])
    
    elif parameters["format"] == "rss":
        reader = RssReader()
        documents = reader.load_data(
            [
                "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
                "https://roelofjanelsinga.com/atom.xml",
            ]
        )
    
    elif parameters["format"] == "whole_site":
        if parameters["url_whole"] is None or parameters["prefix"] is None:
            raise Exception(status_code=400, detail="Base URL and prefix must be provided for Whole Site Reader")
        scraper = WholeSiteReader(prefix=parameters["prefix"], max_depth=parameters["max_depth"])
        documents = scraper.load_data(base_url=parameters["url_whole"])
    
    elif parameters["format"] in ["pdf", "docx", "txt", "image", "ipynb", "pptx", "csv", "xml"]:
        if parameters["url_file"] is None:
            raise Exception(status_code=400, detail="URL must be provided for the selected format")
        if not os.path.exists(parameters["url_file"]):
            raise Exception(status_code=400, detail="File not found")
        
        parser_map = {
            "pdf": PyMuPDFReader(),
            "docx": DocxReader(),
            "txt": FlatReader(),
            "image": ImageReader(),
            "ipynb": IPYNBReader(),
            "pptx": PptxReader(),
            "csv": CSVReader(),
            "xml": XMLReader()
        }
        
        parser = parser_map.get(parameters["format"])
        documents = parser.load_data(parameters["url_file"])
    
    task_data_name = parameters["task_data_management_collection"]
    
    redis_client = redis.Redis(host=parameters["taskHost"], port=parameters["taskPort"], db=0)

    task_id = create_task("Loading documents")
    nodes = []  # Initialize nodes variable
    try:
        nodes = ingestion(documents, task_id, parameters)
        update_task_status(task_id, "completed")
    except Exception as e:
        update_task_status(task_id, f"failed: {str(e)}")
        print(e)
    
    return nodes
     
vector_store=MilvusVectorStore(
    uri=parameters["milvusURI"],  
    dim=parameters["milvusDimension"],  
    overwrite=parameters["milvusOverwrite"],
    collection_name=parameters["milvusCollectionName"],
    consistency_level=parameters["milvusConsistencyLevel"],
    enable_sparse=parameters["milvusEnableSparse"],
    similarity_metric=parameters["milvusSimilarityMetric"],
    embedding_field=parameters['milvusEmbeddingField'],
    doc_id_field=parameters['milvusDocIdField'],
    text_key=parameters['milvusTextKey'],
    sparse_embedding_field=parameters['milvusSparseEmbeddingField'],
    hybrid_ranker="RRFRanker",
    hybrid_ranker_params={"k": 60},
    scalar_field_names=['text1','file_size'],
    scalar_field_types=[DataType.VARCHAR,DataType.INT64],
    batch_size=parameters['milvusBatchSize'],
    index_management=IndexManagement.CREATE_IF_NOT_EXISTS
)

cache=IngestionCache(
    cache=RedisCache.from_host_and_port(host=parameters['cacheHost'], port=parameters['cachePort']),
    collection=parameters['cacheCollectionName'],
)

docstore=RedisDocumentStore.from_host_and_port(
    parameters['docHost'], parameters['docPort'], namespace=parameters['docCollectionName']
)

embed_model=TextEmbeddingsInference(
    model_name=parameters['modelName'],
    base_url=parameters['modelBaseUrl'],
    embed_batch_size=parameters['modelEmbedBatchSize'],
    timeout= parameters['modelTimeout'],
    truncate_text=parameters['modelTruncateText'],
)

def ingestion(documents,load_task,parameters):
    task_id = create_task("Running ingestion pipeline")
    try:
        pipeline = IngestionPipeline(
            transformations=[
                # SentenceSplitter(chunk_size=100, chunk_overlap=0),
                SentenceSplitter(),
                # HuggingFaceEmbedding(model_name=parameters['modelName']),
                embed_model,
            ],
            docstore=docstore,
            cache=cache, 
            vector_store=vector_store
        )

        nodes = pipeline.run(documents=documents)
        update_task_status(task_id, "completed")

    except Exception as e:
        update_task_status(task_id, f"failed: {str(e)}")
        print(f"Pipeline Error: {e}")
        raise Exception(status_code=500, detail=f"Pipeline Error: {e}")
    return {"insert_results": len(nodes),"Pipeline Task ID":task_id,"Load Task ID":load_task}

def print_milvus_contents(collection_name: str):
    try:
        # Connect to Milvus server
        connections.connect("default", uri="http://localhost:19530")
        
        # Check if the collection exists
        if not utility.has_collection(collection_name):
            print(f"Collection '{collection_name}' not found.")
            return

        # Load the collection
        collection = Collection(name=collection_name)

        # Retrieve all data from the collection
        data = collection.query(expr="*", output_fields=["doc_id", "embedding", "text_key"], limit=10)
        
        print(f"\n--- Contents of Milvus Collection: {collection_name} ---")
        for idx, doc in enumerate(data):
            print(f"\nDocument {idx + 1}:")
            print(f"ID: {doc['doc_id']}")
            print(f"Text Key: {doc['text_key']}")
            print(f"Embedding (first 10 dims): {doc['embedding'][:10]}")
    except Exception as e:
        print(f"Error querying Milvus: {e}")

if st.sidebar.button("Ingest Data"):
    try:
        st.write(read_data())
        st.info("Data Ingestion Completed")
    except Exception as e:
        st.error(f"Error: {e}")

if st.sidebar.button("Show All Tasks"):
    try:
        tasks = get_all_tasks()  # Retrieve all tasks
        if tasks:
            # Convert tasks dictionary to a list of rows for display
            task_list = [{"Task ID": task_id, "Description": task_info["description"], "Status": task_info["status"]} 
                         for task_id, task_info in tasks.items()]
            st.info("Tasks Status")
            st.table(task_list)  # Display tasks in a table
        else:
            st.info("No tasks found.")
    except Exception as e:
        st.error(f"Error fetching tasks: {e}")

# Chat history stored in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f'<div style="text-align: left; padding: 10px; background-color: #f0f0f0; border-radius: 10px; margin-left: 160px; margin-bottom: 5px;">'
            # f'<strong>You:</strong> {message["content"]}</div>',
            f'{message["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="text-align: left; padding: 10px; background-color: #e8fbe8; border-radius: 10px; margin-right: 160px; margin-bottom: 5px;">'
            # f'<strong>Assistant:</strong> {message["content"]}</div>',
            f'{message["content"]}</div>',
            unsafe_allow_html=True,
        )
# fbe8e8
# # Display chat messages in the UI
# for message in st.session_state.messages:
#     if message["role"] == "user":
#         st.markdown(f"**You:** {message['content']}")
#     else:
#         st.markdown(f"**Assistant:** {message['content']}")
        
# Input field for user query
query_str = st.text_input("Ask a question: ", "")

if st.button("Ask"):
    if query_str:
        st.session_state.messages.append({"role": "user", "content": query_str})
    
        query_embedding = embed_model.get_query_embedding(query_str)

        query_mode = "default"
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
        )

        vector_store = MilvusVectorStore(
            uri="http://localhost:19530",  # Adjust as needed
            dim=1024,  # Adjust dimension based on your embedding model
            overwrite=False,
            collection_name="app_milvus_db",
            consistency_level="Eventually",
        )

        query_result = vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        class VectorDBRetriever(BaseRetriever):
            """Retriever over a milvus vector store."""

            def __init__(
                self,
                vector_store: MilvusVectorStore,
                embed_model: Any,
                query_mode: str = "default",
                similarity_top_k: int = 2,
            ) -> None:
                """Init params."""
                self._vector_store = vector_store
                self._embed_model = embed_model
                self._query_mode = query_mode
                self._similarity_top_k = similarity_top_k
                super().__init__()

            def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
                """Retrieve."""
                query_embedding = embed_model.get_query_embedding(
                    query_bundle.query_str
                )
                vector_store_query = VectorStoreQuery(
                    query_embedding=query_embedding,
                    similarity_top_k=self._similarity_top_k,
                    mode=self._query_mode,
                )
                query_result = vector_store.query(vector_store_query)

                nodes_with_scores = []
                for index, node in enumerate(query_result.nodes):
                    score: Optional[float] = None
                    if query_result.similarities is not None:
                        score = query_result.similarities[index]
                    nodes_with_scores.append(NodeWithScore(node=node, score=score))

                return nodes_with_scores
            
        retriever = VectorDBRetriever(
            vector_store, embed_model, query_mode="default", similarity_top_k=2
        )

        llm=Ollama(model=parameters["selected_model"], request_timeout=300.0)
        
        from llama_index.core.query_engine import RetrieverQueryEngine

        query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

        response = query_engine.query(query_str)

        st.session_state.messages.append({"role": "assistant", "content": str(response)})
    

