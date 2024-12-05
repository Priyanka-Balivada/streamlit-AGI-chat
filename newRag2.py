import logging
from langchain_milvus import Milvus
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from text_generation import Client
from pymilvus import connections, utility
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define RAG prompt template
rag_prompt_intel_raw = """### System: You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

### User: Question: {question}
Context: {context}
### Assistant: """

# Initialize embeddings and Milvus store
embeddings = HuggingFaceHubEmbeddings(model="http://localhost:8081", huggingfacehub_api_token="EMPTY")
collection_name = "rag_new_milvus_db"
store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": "http://localhost:19530"},
    collection_name=collection_name,
    vector_field="embedding",
    auto_id=True,
    drop_old=False
)

# Function to retrieve sources
def get_sources(question):
    try:
        # Connect to Milvus server
        connections.connect(uri="http://localhost:19530")
        
        # Check if the collection exists
        if not utility.has_collection(collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return []
        
        # Perform similarity search
        return store.similarity_search(
            f"Represent this sentence for searching relevant passages: {question}", k=2
        )
    except Exception as e:
        print(f"Error in get_sources: {e}")
        return []

# Function to convert sources into a readable string
def sources_to_str(sources):
    return "\n".join(f"{i+1}. {s.page_content}" for i, s in enumerate(sources))

# Function to generate answers using TGI
def get_answer(question, sources):
    try:
        client = Client("http://localhost:8084")  # Adjust port as needed
        context = "\n".join(s.page_content for s in sources)
        prompt = rag_prompt_intel_raw.format(question=question, context=context)
        
        # Generate response
        return client.generate(
            prompt, max_new_tokens=512, stop_sequences=["### User:", "</s>"]
        ).generated_text
    except Exception as e:
        logging.error(f"Error in get_answer: {e}")
        return "An error occurred while generating the answer."

# Function to handle the RAG process
def rag_answer(question):
    # Retrieve sources
    sources = get_sources(question)
    
    if not sources:
        return "No relevant sources found. Please try a different question."
    
    # Generate answer from sources
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
        # Add user query to session state
        st.session_state.messages.append({"role": "user", "content": query_str})
        
        # Generate response using RAG pipeline
        with st.spinner("Generating answer..."):
            response = rag_answer(query_str)
        
        # Add assistant's response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat history (without duplication)
for i in range(0, len(st.session_state.messages), 2):
    # Ensure even pairing of user and assistant
    user_message = st.session_state.messages[i]
    assistant_message = st.session_state.messages[i + 1] if i + 1 < len(st.session_state.messages) else None

    # Display user query
    st.markdown(f"{user_message['content']}")
    # Display assistant response
    if assistant_message:
        st.markdown(f"{assistant_message['content']}")
