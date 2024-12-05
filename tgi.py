from Text import TextGenerationInference

llm = TextGenerationInference(
        model_name="HuggingFaceH4/zephyr-7b-beta",  # Ensure the model name is correct
        model_url="http://127.0.0.1:8084",
        token="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"
        )
prompt = "Hello, how are you?"
response = llm.complete("Hello, how are you?")
print(response)