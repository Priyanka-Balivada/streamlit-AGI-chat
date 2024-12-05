from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from llama_index.llms.langchain import LangChainLLM

client = HuggingFaceEndpoint(
    endpoint_url="http://localhost:8084/",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)

llm = LangChainLLM(llm=client)

# Use the `invoke` method instead of `complete`
response_gen = llm.complete("Capital of India?")
print(f"Response: {str(response_gen)}")

from langchain_community.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference

# Basic Example (no streaming)
llm = HuggingFaceTextGenInference(
    inference_server_url="http://localhost:8084/",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)
print(llm.invoke("What is Capital of India?"))  # noqa: T201
