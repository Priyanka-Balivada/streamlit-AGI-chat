from pymilvus import connections
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_milvus import Milvus
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.prompts import PromptTemplate
import logging

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

collection_name = "app_milvus_db"

# Initialize the embeddings model

from langchain_community.embeddings import HuggingFaceHubEmbeddings
embeddings = HuggingFaceHubEmbeddings(model="http://localhost:8081", huggingfacehub_api_token="EMPTY")

# Configure Milvus as the vector database
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": "http://localhost:19530"},
    collection_name=collection_name,
    vector_field="embedding",
    auto_id=True,
    drop_old=False
)

# Set up the retriever
retriever = VectorStoreRetriever(
    vectorstore=vector_store,
    search_type="similarity",
    embeddings=embeddings
)

# Initialize the HuggingFace Endpoint for the LLM
llm = HuggingFaceEndpoint(
    endpoint_url="http://localhost:8084/",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)

# Set up the retrieval pipeline
retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=retriever)

# Query Milvus and generate a response
query = "What is the scaling factor"

# Add logging to ensure the retriever is being used correctly
logging.basicConfig(level=logging.DEBUG)

# Retrieve documents from Milvus using the correct method
# retrieved_docs = retriever.get_relevant_documents(query)  # Use get_relevant_documents instead of retrieve
# print("Retrieved documents:", retrieved_docs)

# Invoke the retrieval QA chain
response = retrievalQA.invoke(query)
print("Response:", response)
