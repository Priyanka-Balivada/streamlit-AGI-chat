import logging
from langchain_milvus import Milvus
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from text_generation import Client
from pymilvus import connections, utility
import streamlit as st
import json
from pymilvus import MilvusClient
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.vector_stores.milvus import MilvusVectorStore

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define RAG prompt template
rag_prompt_intel_raw = """### System: You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

### User: Question: {question}

Context: {context}

### Assistant: """

# Initialize embeddings and Milvus store
embeddings = HuggingFaceHubEmbeddings(model="http://localhost:8081", huggingfacehub_api_token="EMPTY")
collection_name = "app_milvus_db"
store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": "http://localhost:19530"},
    collection_name=collection_name,
    vector_field="embedding",
    auto_id=True,
    drop_old=False,
)


# Function to retrieve sources
# def get_sources(question):
#     try:
#         connections.connect(uri="http://localhost:19530")
#         if not utility.has_collection(collection_name):
#             logging.info(f"Collection '{collection_name}' does not exist.")
#             return []
#         return store.similarity_search(
#             f"{question}", k=2
#         )
#     except Exception as e:
#         logging.error(f"Error in get_sources: {e}")
#         return []

def get_sources(question):
    try:
        local_embed_model = TextEmbeddingsInference(
            model_name="BAAI/bge-large-en-v1.5",
            base_url="http://127.0.0.1:8081",  # Adjust this URL to your inference server
            timeout=60,
            embed_batch_size=10,
        )
        # Connect to Milvus
        client = MilvusClient(uri="http://localhost:19530")

        embeddings = local_embed_model.get_query_embedding(question)
        
        # Perform search
        res = client.search(
            collection_name="app_milvus_db",  # Replace with your collection name
            data=[embeddings],
            limit=1,  # Number of results
            output_fields=["_node_content"],  # Replace with a valid field name, e.g., 'text' or 'metadata'
        )

        result=""
        for search_result in res[0]:  # Assuming `res` is a list containing the first batch of results
            val=search_result.get('entity', None)
            if val != None:
                val2 = val.get('_node_content', None)
                if val2 != None:
                    if isinstance(val2, dict):
                        # If it's a dictionary, safely access the "text" field
                        text_content = val2.get("text", "No text available")
                    else:
                        # If it's a string, assume it's the text content
                        text_content = val2 if val2 else "No text available"

                        # Print the content
                        # print("Doc")
                        # print(text_content)

                        parsed_content = json.loads(text_content)

                        text_content = parsed_content.get("text", "No text available")

                        # print(text_content)

                        result+=text_content+"\n\n"
        if result:
            return result
        else:
            return None
    except Exception as e:
        logging.error(f"Error in get_sources: {e}")
        return []


# Function to convert sources into a readable string
def sources_to_str(sources):
    return "\n".join(f"{i+1}. {s.page_content}" for i, s in enumerate(sources))

# Function to generate answers using TGI
def get_answer(question, sources):
    try:
        client = Client("http://localhost:8084")  # Adjust port as needed
        # context = "\n".join(s.page_content for s in sources)
        context=sources
        prompt = rag_prompt_intel_raw.format(question=question, context=context)
        return client.generate(
            prompt, max_new_tokens=512, stop_sequences=["### User:", "</s>"]
        ).generated_text
    except Exception as e:
        logging.error(f"Error in get_answer: {e}")
        return "An error occurred while generating the answer."

# Function to handle the RAG process
def rag_answer(question):
    sources = get_sources(question)
    if not sources:
        return "No relevant sources found. Please try a different question."
    answer = get_answer(question, sources)
    return answer

# Streamlit App
st.title("Intel Gaudi 2 RAG App")
st.write("Ask questions to the RAG-powered system.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Query input
query_str = st.text_input("Ask a question: ", "")

# Ask button functionality
if st.button("Ask"):
    if query_str:
        # Add user question to session state
        st.session_state.messages.append({"role": "user", "content": query_str})
        
        # Generate response using RAG pipeline
        with st.spinner("Generating answer..."):
            response = rag_answer(query_str)
            print(response)
        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.write(f"**User:** {message['content']}")
            else:
                st.write(f"**Assistant:** {message['content']}")
