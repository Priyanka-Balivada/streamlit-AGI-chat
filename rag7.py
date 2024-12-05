import streamlit as st
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from typing import List, Optional
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List
# Initialize Streamlit interface
st.title("RAG Chatbot Interface")
st.write("Ask me anything, and I'll fetch relevant information from the document store.")


# Embedding model setup
embed_model = TextEmbeddingsInference(
    model_name="BAAI/bge-large-en-v1.5",
    base_url="http://127.0.0.1:8081",
    timeout=60,
    embed_batch_size=10,
)

# Set up the Milvus vector store connection
vector_store = MilvusVectorStore(
    uri="http://localhost:19530",
    dim=384,
    overwrite=False,
    collection_name="app_milvus_db",
    consistency_level="Eventually",
    enable_sparse=True,
)

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


# Chat history stored in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input field for user query
user_input = st.text_input("You: ", "")

# Action when the 'Ask' button is clicked
if st.button("Ask"):
    if user_input:
        # Display user message in chat history
        st.session_state.messages.append({"role": "user", "content": user_input})


        # Generate query embedding
        query_embedding = embed_model.get_query_embedding(user_input)

        # Set query parameters (default retrieval mode)
        query_mode = "default"
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=2,
            mode=query_mode,
        )

        

        # Perform vector store query
        query_result = vector_store.query(vector_store_query)

        # Process retrieved nodes with scores
        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        # Create retriever and query engine
        retriever = VectorDBRetriever(
            vector_store, embed_model, query_mode="default", similarity_top_k=2
        )

        # Set up the LLM model for response generation
        llm = Ollama(model="llama3.2", request_timeout=300.0)

        query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

        # Query the model
        response = query_engine.query(user_input)

        # Display retrieved information and response in chat
        st.session_state.messages.append({"role": "assistant", "content": str(response)})
    
# Display chat messages in the UI
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Assistant:** {message['content']}")

