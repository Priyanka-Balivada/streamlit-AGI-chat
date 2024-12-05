
parameters = {
    #content ingestion
    "format": st.selectbox("Format", ["web", "whole_site","pdf", "docx", "txt", "image", "ipynb", "pptx", "csv", "xml","rss","beautiful_soup","html_tags"]),

    #task management configurations
    "task_data_management_collection": st.sidebar.text_input("Task Management Collection Name", value="task_data"),
    "taskHost": st.sidebar.text_input("Task Management Host", value="localhost"),
    "taskPort": st.sidebar.number_input("Task Management Port", value=6379),

    #embedding model configurations
    "modelName": st.sidebar.text_input("Model Name", value="BAAI/bge-small-en-v1.5"),
    "modelBaseUrl": st.sidebar.text_input("Model Base URL", value="http://127.0.0.1:8081"),
    "modelEmbedBatchSize": st.sidebar.number_input("Model Embed Batch Size", value=10),
    "modelTimeout": st.sidebar.number_input("Model Timeout",step=1.,format="%.2f",value=60.0),
    "modelTruncateText": st.sidebar.checkbox("Model Truncate Text", value=True),
    
    #docstore configurations
    "docHost": st.sidebar.text_input("Doc Host", value="localhost"),
    "docPort": st.sidebar.number_input("Doc Port", value=6379),
    "docCollectionName": st.sidebar.text_input("Document Collection Name", value="document_store"),
    
    #cache configurations
    "cacheHost": st.sidebar.text_input("Cache Host", value="127.0.0.1"),
    "cachePort": st.sidebar.number_input("Cache Port", value=6379),
    "cacheCollectionName": st.sidebar.text_input("Cache Collection Name", value="cache"),
    
    #vectorstore configurations
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
    "milvusBatchSize": st.sidebar.number_input("Milvus Batch Size", value=100)
}

if parameters["format"] == "web":
    parameters["url_web"]=st.text_input("URL Web", value=None)
elif parameters["format"] == "html_tags":
    parameters["url_html"]=st.text_input("URL HTML", value=None)
    parameters["tag"]=st.text_input("Tag", value="section"),
    parameters["ignore_no_id"]=st.checkbox("Ignore No Id", value=True)
elif parameters["format"] == "beautiful_soup":
    parameters["url_soup"]=st.text_input("URL Soup", value=None)
elif parameters["format"] == "whole_site":
    parameters["url_whole"]=st.text_input("URL Whole", value=None)
    parameters["prefix"]=st.text_input("Prefix", value=None)
    parameters["max_depth"]=st.number_input("Max Depth", value=10)
elif parameters["format"] in ["pdf", "docx", "txt", "image", "ipynb", "pptx", "csv", "xml"]:
    parameters["url_file"]=st.text_input("URL File", value=None)

if parameters["format"] == "web":
    parameters["url_web"] = st.text_input("URL", value=None)
elif parameters["format"] == "html_tags":
    parameters["url_html"] = st.text_input("HTML URL", value=None)
    parameters["tag"] = st.text_input("Tag", value="section")
    parameters["ignore_no_id"] = st.checkbox("Ignore No ID", value=True)
elif parameters["format"] == "beautiful_soup":
    parameters["url_soup"] = st.text_input("Soup URL", value=None)
elif parameters["format"] == "whole_site":
    parameters["url_whole"] = st.text_input("Base URL", value=None)
    parameters["prefix"] = st.text_input("Prefix", value=None)
    parameters["max_depth"] = st.number_input("Max Depth", value=10)
elif parameters["format"] in ["pdf", "docx", "txt", "image", "ipynb", "pptx", "csv", "xml"]:
    parameters["url_file"] = st.text_input("File Path", value=None)


import subprocess
import streamlit as st
from llama_index.llms.ollama import Ollama

# Function to get installed Ollama models using the 'ollama list' command
def get_ollama_models():
    try:
        # Execute the 'ollama list' command and capture the output
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        
        # Parse the output (assuming the models are listed line by line)
        models = result.stdout.splitlines()
        
        # Filter out empty lines or unnecessary information (if needed)
        models = [model for model in models if model.strip()]
        return models
    except subprocess.CalledProcessError as e:
        st.error(f"Error occurred while trying to fetch models: {e}")
        return []

# Get the available models using 'ollama list'
available_models = get_ollama_models()

# Check if models are available
if available_models:
    # Dropdown to select the model
    selected_model = st.selectbox("Select an Ollama Model", available_models)
    st.write(f"You selected: {selected_model}")

    # Set up the LLM model based on the selected model
    llm = Ollama(model=selected_model, request_timeout=300.0)

else:
    st.write("No models found. Please ensure you have models installed using 'ollama list'.")

# Example usage of selected model for a query (if needed):
if st.button("Ask"):
    # Example query
    query = "What is the capital of France?"
    response = llm.query(query)  # Make sure `llm.query()` is the right method
    st.write(f"Response from model: {response}")
