# Hybrid Retrieval

from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeWithScore
from typing import Optional, Any, List, Dict
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.query_engine import RetrieverQueryEngine

# Initialize the embedding model
print("Import")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
print("TEI")

# Set query string
query_str = "what are different components?"
query_embedding = embed_model.get_query_embedding(query_str)

query_mode = "hybrid"  # Default, Sparse, or Hybrid

# Define Vector Store Query
vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding,
    query_str=query_str,
    similarity_top_k=2,
    sparse_top_k=3,
    hybrid_top_k=2,
    alpha=0.5,  # Adjust alpha to balance between sparse and dense scores
    mode=query_mode
)
print("Vector Query")

# Configure the Vector Store
vector_store = MilvusVectorStore(
    uri="http://localhost:19530",  # Adjust as needed
    collection_name="app_milvus_db",
    enable_sparse=True
)
print("Milvus")

# Execute Hybrid Query
query_result = vector_store.query(vector_store_query)

print("\n\n\nVector Query\n\n")
if query_result.nodes:
    print(query_result.nodes[0].get_content())


# Define the Retriever Class
class VectorDBRetriever(BaseRetriever):
    """Retriever supporting hybrid queries over a Milvus vector store."""

    def __init__(
        self,
        vector_store: MilvusVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
        sparse_top_k: int = 3,
        hybrid_top_k: int = 2,
        alpha: float = 0.5,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        self._sparse_top_k = sparse_top_k
        self._hybrid_top_k = hybrid_top_k
        self._alpha = alpha
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve using hybrid search."""
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            query_str=query_bundle.query_str,
            similarity_top_k=self._similarity_top_k,
            sparse_top_k=self._sparse_top_k,
            hybrid_top_k=self._hybrid_top_k,
            alpha=self._alpha,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores


# Initialize the Retriever with hybrid mode
retriever = VectorDBRetriever(
    vector_store,
    embed_model,
    query_mode="hybrid",
    similarity_top_k=2,
    sparse_top_k=3,
    hybrid_top_k=2,
    alpha=0.5
)

# Initialize LlamaCPP
model_url = model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
llm = LlamaCPP(
    model_url=model_url,
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    model_kwargs={"n_gpu_layers": 1},
    verbose=True,
)

# Build the Query Engine
query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

# Execute the Query
response = query_engine.query(query_str)

print("\n\n\nRetrieval Query\n")
print(str(response))
