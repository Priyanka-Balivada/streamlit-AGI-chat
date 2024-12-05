from pymilvus import connections, Collection
# from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_milvus import Milvus
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceHubEmbeddings

# Connect to Milvus
# connections.connect("default", host="localhost", port="19530")
connections.connect("default", host="localhost", port="19530")
collection_name = "app_db_milvus"


# Initialize the embeddings model
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

# retriever = VectorStoreRetriever(vectorstore=vector_store)
retriever = VectorStoreRetriever(
    vectorstore=vector_store,
    search_type="similarity",
    # search_kwargs={"metric_type": "IP", "params": {}, "k": 5}
)

# Initialize the HuggingFace Endpoint
llm = HuggingFaceEndpoint(
    endpoint_url="http://localhost:8084/",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)

# # Set up the retrieval pipeline
# retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 similar documents

# # Define the prompt template for the QA chain
# prompt_template = PromptTemplate(
#     template=(
#         "Use the following pieces of context to answer the question at the end. "
#         "If you don't know the answer, just say you don't know. "
#         "Don't try to make up an answer.\n\n"
#         "{context}\n\nQuestion: {question}\nAnswer:"
#     ),
#     input_variables=["context", "question"],
# )

# # # Set up the QA chain
# qa_chain = RetrievalQA(
#     retriever=retriever,
#     llm=llm,
#     prompt_template=prompt_template,
# )


# ****************************************************************************
retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=retriever)

# # # Query Milvus and generate a response
query = "What are different components?"
response = retrievalQA.invoke(query)

# # response=qa_chain.invoke(query)
# print("\n\n\n")
print(response)


# ********************************************************************************

# ********************************************************************************
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate

# system_prompt = (
#     "Use the given context to answer the question. "
#     "If you don't know the answer, say you don't know. "
#     "Use three sentence maximum and keep the answer concise. "
#     "Context: {context}"
# )
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )
# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# chain = create_retrieval_chain(retriever, question_answer_chain)

# print("\n\n\n\n")
# res=chain.invoke({"input": query})

# print(res["answer"])


# ***********************************************************************************

# from langchain.retrievers.multi_query import MultiQueryRetriever

# question = "What are different components?"
# # llm = ChatOpenAI(temperature=0)
# retriever_from_llm = MultiQueryRetriever.from_llm(
#     retriever=retriever, llm=llm
# )

# unique_docs = retriever_from_llm.invoke(question)
# print("\n\n\n")
# print(unique_docs)