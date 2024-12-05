#default retrieval
import streamlit as st
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
# construct vector store query
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.milvus import MilvusVectorStore

query_str=st.text_input("Ask")
if st.button("Ask"):
    print("Import")
    embed_model = TextEmbeddingsInference(
        model_name="BAAI/bge-large-en-v1.5",
        base_url="http://127.0.0.1:8081",
        timeout=60,  # timeout in seconds
        embed_batch_size=10,  # batch size for embedding
    )
    print("TEI")

    # query_str = "Can you tell me about LlamaIndex?"
    # query_str = "Can you tell me about Big AGI Interface?"
    # query_str = "Hello, who are you?"
    query_embedding = embed_model.get_query_embedding(query_str)

    query_mode = "default"
    # query_mode = "sparse"
    # query_mode = "hybrid"

    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
    )
    print("Vector Query")
    # returns a VectorStoreQueryResult

    vector_store = MilvusVectorStore(
        uri="http://localhost:19530",  # Adjust as needed
        dim=384,  # Adjust dimension based on your embedding model
        overwrite=False,
        collection_name="app_milvus_db",
        consistency_level="Eventually",
        enable_sparse=True,
    )
    print("Milvus")

    query_result = vector_store.query(vector_store_query)

    print("\n\n\nVector Query\n\n")
    for i in query_result.nodes:
        print(i.get_content())


    from llama_index.core.schema import NodeWithScore
    from typing import Optional

    nodes_with_scores = []
    for index, node in enumerate(query_result.nodes):
        score: Optional[float] = None
        if query_result.similarities is not None:
            score = query_result.similarities[index]
        nodes_with_scores.append(NodeWithScore(node=node, score=score))

    from llama_index.core import QueryBundle
    from llama_index.core.retrievers import BaseRetriever
    from typing import Any, List


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

    from llama_index.llms.ollama import Ollama
    llm=Ollama(model="llama3.2", request_timeout=300.0)
    # from llama_index.llms.llama_cpp import LlamaCPP

    # # model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin"
    # # model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
    # model_url = "https://huggingface.co/igorbkz/gpt2-Q8_0-GGUF/resolve/main/gpt2.Q8_0.gguf"
    # llm = LlamaCPP(
    #     # You can pass in the URL to a GGML model to download it automatically
    #     model_url=model_url,
    #     # optionally, you can set the path to a pre-downloaded model instead of model_url
    #     model_path=None,
    #     temperature=0.1,
    #     max_new_tokens=256,
    #     # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    #     context_window=3900,
    #     # kwargs to pass to __call__()
    #     generate_kwargs={},
    #     # kwargs to pass to __init__()
    #     # set to at least 1 to use GPU
    #     model_kwargs={"n_gpu_layers": 1},
    #     verbose=True,
    # )

    from llama_index.core.query_engine import RetrieverQueryEngine

    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

    query_str1 = "What are different components of AGI Chat Interface pipeline?"
    query_str2 = "Hello, who are you?"
    response = query_engine.query(query_str2)

    print("\n\n\nRetrieval Query\n")
    print(str(response))
    print(response.source_nodes[0].get_content())