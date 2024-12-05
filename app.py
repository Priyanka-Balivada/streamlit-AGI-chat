import streamlit as st
import redis
import uuid
import json
from typing import Any, Optional
from llama_index.readers.web import (
    SimpleWebPageReader,
    BeautifulSoupWebReader,
    RssReader,
    WholeSiteReader,
)
from llama_index.readers.file import (
    DocxReader,
    PyMuPDFReader,
    FlatReader,
    ImageReader,
    IPYNBReader,
    PptxReader,
    CSVReader,
    XMLReader,
)
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.vector_stores.milvus import MilvusVectorStore, IndexManagement
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.core.node_parser import SentenceSplitter
from pymilvus import connections, utility, DataType, Collection
from llama_index.storage.docstore.redis import RedisDocumentStore

# Redis client global variables
redis_client = None
task_data_name = ""

# Streamlit UI
st.title("Document Ingestion and Vector Store Management")

# Sidebar for Redis and vector store configurations
st.sidebar.header("Configuration")
task_data_name = st.sidebar.text_input("Task Data Collection Name", value="task_data")
redis_host = ""
redis_port = 6379

# Get model parameters from user
parameters = {
    "modelName": st.sidebar.text_input("Model Name", value="BAAI/bge-small-en-v1.5"),
    "modelBaseUrl": st.sidebar.text_input("Model Base URL", value="http://127.0.0.1:8081"),
    "modelEmbedBatchSize": st.sidebar.number_input("Model Embed Batch Size", value=64),
    "modelTimeout": st.sidebar.number_input("Model Timeout", value=30),
    "modelTruncateText": st.sidebar.checkbox("Model Truncate Text", value=True),
    
    "docHost": st.sidebar.text_input("Doc Host", value="localhost"),
    "docPort": st.sidebar.number_input("Doc Port", value=6379),
    "docCollectionName": st.sidebar.text_input("Document Collection Name", value="document_store"),
    
    "cacheHost": st.sidebar.text_input("Cache Host", value="localhost"),
    "cachePort": st.sidebar.number_input("Cache Port", value=6379),
    "cacheCollectionName": st.sidebar.text_input("Cache Collection Name", value="cache"),
    
    "milvusURI": st.sidebar.text_input("Milvus URI", value="http://localhost:19530"),
    "milvusDimension": st.sidebar.number_input("Milvus Dimension", value=1024),
    "milvusOverwrite": st.sidebar.checkbox("Milvus Overwrite", value=False),
    "milvusCollectionName": st.sidebar.text_input("Milvus Collection Name", value="app_milvus_db"),
    "milvusConsistencyLevel": st.sidebar.selectbox("Milvus Consistency Level", ["Eventually", "Strong","Bounded","Session"]),
    "milvusEnableSparse": st.sidebar.checkbox("Milvus Enable Sparse", value=False),
    "milvusSimilarityMetric": st.sidebar.selectbox("Milvus Similarity Metric", ["IP", "L2"]),
    "milvusEmbeddingField": st.sidebar.text_input("Milvus Embedding Field", value="embedding"),
    "milvusDocIdField": st.sidebar.text_input("Milvus Doc ID Field", value="doc_id"),
    "milvusTextKey": st.sidebar.text_input("Milvus Text Key", value="text_key"),
    "milvusSparseEmbeddingField": st.sidebar.text_input("Milvus Sparse Embedding Field", value="sparse_embedding"),
    "milvusBatchSize": st.sidebar.number_input("Milvus Batch Size", value=100),
}

redis_host=parameters["docHost"]
redis_port=parameters['docPort']

# Connect to Redis
try:
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
    st.sidebar.success("Connected to Redis")
except Exception as e:
    st.sidebar.error(f"Redis connection failed: {e}")

# Main document ingestion options
format_option = st.selectbox("Select Document Format", ["web", "html_tags", "pdf", "docx", "rss"])
url_input = st.text_input("Document URL or File Path", value="")
additional_config = {}

# Take additional inputs based on the document format selected
if format_option == "html_tags":
    additional_config["tag"] = st.text_input("HTML Tag", value="section")
    additional_config["ignore_no_id"] = st.checkbox("Ignore No ID", value=True)
elif format_option == "rss":
    url_input = st.text_area("RSS URLs (comma-separated)", value="https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml")


# Create a task in Redis
def create_task(description: str) -> str:
    global redis_client, task_data_name
    task_id = str(uuid.uuid4())
    task_data = {"description": description, "status": "in process"}
    redis_client.hset(task_data_name, task_id, json.dumps(task_data))
    return task_id

# Update the task status in Redis
def update_task_status(task_id: str, status: str):
    if redis_client.hexists(task_data_name, task_id):
        task_data = json.loads(redis_client.hget(task_data_name, task_id))
        task_data["status"] = status
        redis_client.hset(task_data_name, task_id, json.dumps(task_data))

# Ingest documents into vector store
def ingestion(documents, task_id, parameters):
    try:
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(),
                TextEmbeddingsInference(
                    model_name=parameters["modelName"],
                    base_url=parameters["modelBaseUrl"],
                    embed_batch_size=parameters["modelEmbedBatchSize"],
                    timeout=parameters["modelTimeout"],
                    truncate_text=parameters["modelTruncateText"],
                ),
            ],
            docstore=RedisDocumentStore.from_host_and_port(
                parameters["docHost"], parameters["docPort"], namespace=parameters["docCollectionName"]
            ),
            cache=IngestionCache(
                cache=RedisCache.from_host_and_port(
                    host=parameters["cacheHost"], port=parameters["cachePort"]
                ),
                collection=parameters["cacheCollectionName"],
            ),
            vector_store=MilvusVectorStore(
                uri=parameters["milvusURI"],
                dim=parameters["milvusDimension"],
                overwrite=parameters["milvusOverwrite"],
                collection_name=parameters["milvusCollectionName"],
                consistency_level=parameters["milvusConsistencyLevel"],
                enable_sparse=parameters["milvusEnableSparse"],
                similarity_metric=parameters["milvusSimilarityMetric"],
                embedding_field=parameters["milvusEmbeddingField"],
                doc_id_field=parameters["milvusDocIdField"],
                text_key=parameters["milvusTextKey"],
                sparse_embedding_field=parameters["milvusSparseEmbeddingField"],
                batch_size=parameters["milvusBatchSize"],
            ),
        )

        nodes = pipeline.run(documents=documents)
        update_task_status(task_id, "completed")
        return {"insert_results": len(nodes), "Pipeline Task ID": task_id}

    except Exception as e:
        update_task_status(task_id, f"failed: {str(e)}")
        st.error(f"Pipeline Error: {e}")
        return None


if st.button("Run Ingestion"):
    try:
        # Load documents based on format
        documents = []
        if format_option == "web":
            documents = SimpleWebPageReader(html_to_text=True).load_data([url_input])
        elif format_option == "html_tags":
            with open(url_input, "r", encoding="utf-8") as f:
                content = f.read()
            documents = BeautifulSoupWebReader().load_data(content)
        elif format_option == "rss":
            documents = RssReader().load_data(url_input.split(","))
        elif format_option == "pdf":
            documents = PyMuPDFReader().load_data(url_input)
        elif format_option == "docx":
            documents = DocxReader().load_data(url_input)

        task_id = create_task("Document ingestion")
        result = ingestion(documents, task_id, parameters)
        st.success(f"Ingestion completed. Task ID: {task_id}")
        st.write(result)

    except Exception as e:
        st.error(f"Error: {e}")
