# from text_generation import Client
# from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

# llm = HuggingFaceEndpoint(
#     endpoint_url="http://localhost:8084/",
#     max_new_tokens=512,
#     top_k=10,
#     top_p=0.95,
#     typical_p=0.95,
#     temperature=0.01,
#     repetition_penalty=1.03,
#     # huggingfacehub_api_token="my-api-key"
# )
# print(llm.invoke("What is Deep Learning?"))

from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    endpoint_url="http://localhost:8084/",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    # huggingfacehub_api_token="my-api-key"
)
print(llm.invoke("What is Deep Learning?"))
