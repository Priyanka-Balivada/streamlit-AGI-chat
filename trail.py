# gradio working

# Import necessary libraries
import logging
from langchain_milvus import Milvus
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from text_generation import Client
from pymilvus import connections, utility
import gradio as gr
from pathlib import Path

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define the RAG prompt template
rag_prompt_intel_raw = """### System: You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

### User: Question: What are different components of AGI Chat Interface pipeline?

Context: {context}

### Assistant: """

embeddings = HuggingFaceHubEmbeddings(model="http://localhost:8081", huggingfacehub_api_token="EMPTY")
collection_name = "rag_new_milvus_db"
store = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": "http://localhost:19530"},
            collection_name=collection_name,
            vector_field="embedding",
            auto_id=True
        )

def load_file_to_db(path: str, store: Milvus):
    loader = TextLoader(path)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    for chunk in text_splitter.split_documents(document):
        store.add_documents([chunk])

for doc in Path("data/").glob("*.txt"):
    print(f"Loading {doc}...")
    load_file_to_db(str(doc), store)

print("Finished.")

# Function to retrieve sources from Milvus
def get_sources(question):
    try:
        # Establish connection with HuggingFace embeddings model
        logging.info("Connecting to HuggingFace embedding model...")
        embeddings = HuggingFaceHubEmbeddings(
            model="http://localhost:8081", huggingfacehub_api_token="EMPTY"
        )

        # Connect to Milvus database
        logging.info("Connecting to Milvus database...")
        connections.connect(uri="http://localhost:19530")
        collection_name = "rag_new_milvus_db"

        # Check if the collection exists; create if it doesn't
        if not utility.has_collection(collection_name):
            logging.info(f"Collection '{collection_name}' does not exist. Creating it...")
            store = Milvus(
                embedding_function=embeddings,
                connection_args={"uri": "http://localhost:19530"},
                collection_name=collection_name,
                vector_field="embedding",
                auto_id=True
            )
            store.create_collection(
                name=collection_name,
                dimension=384,  # Set the dimension according to your embedding model
                metric="IP",    # Use "IP" (Inner Product) for similarity search
            )
        else:
            logging.info(f"Collection '{collection_name}' already exists.")
            store = Milvus(
                embedding_function=embeddings,
                connection_args={"uri": "http://localhost:19530"},
                collection_name=collection_name,
                vector_field="embedding",
            )

        # Perform similarity search
        logging.info(f"Performing similarity search for: {question}")
        return store.similarity_search(
            f"Represent this sentence for searching relevant passages: {question}",
            k=2,
        )
    except Exception as e:
        logging.error(f"Error in get_sources: {e}")
        return []

# Function to convert sources into a readable string
def sources_to_str(sources):
    return "\n".join(f"{i+1}. {s.page_content}" for i, s in enumerate(sources))

# Function to generate answers using TGI
def get_answer(question, sources):
    try:
        logging.info("Connecting to the TGI endpoint...")
        client = Client("http://localhost:8084")  # Change this to 9009 if using a new model
        context = "\n".join(s.page_content for s in sources)
        prompt = rag_prompt_intel_raw.format(question=question, context=context)

        logging.info("Generating answer using TGI...")
        return client.generate(
            prompt, max_new_tokens=512, stop_sequences=["### User:", "</s>"]
        ).generated_text
    except Exception as e:
        logging.error(f"Error in get_answer: {e}")
        return "An error occurred while generating the answer."

# Default question for the app
default_question = "What is the summary of this document?"

# Function to handle the entire RAG process
def rag_answer(question):
    logging.info("Processing question through RAG pipeline...")
    sources = get_sources(question)
    if not sources:
        return "No relevant sources found. Please try a different question."
    answer = get_answer(question, sources)
    return answer

# Define the Gradio app
with gr.Blocks() as demo:
    gr.Markdown("# Intel Gaudi 2 RAG App")
    question = gr.Textbox(default_question, label="Question")
    answer = gr.Textbox(label="Answer", lines=5)
    send_btn = gr.Button("Run")
    send_btn.click(fn=rag_answer, inputs=question, outputs=answer)

# Launch the app
if __name__ == "__main__":
    logging.info("Launching Gradio app...")
    demo.launch(server_port=7860)  # Modify the port if needed
