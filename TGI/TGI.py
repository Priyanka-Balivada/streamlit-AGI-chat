import requests
from packaging import version
from typing import Sequence, Union, List, Optional, Dict, Any
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, CompletionResponse
from text_generation import Client as TGIClient

class TextGenerationInference:
    def __init__(
        self,
        model_url: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 100,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        num_return_sequences: Optional[int] = None,
        token: Optional[str] = None,
        timeout: float = 120,
        max_retries: int = 5,
        cookies: Optional[Dict[str, str]] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_url = model_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.num_return_sequences = num_return_sequences
        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries
        self.cookies = cookies
        self.additional_kwargs = additional_kwargs or {}

        # Set up headers
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

        # Initialize client
        self.client = TGIClient(base_url=self.model_url, headers=self.headers, cookies=self.cookies)

    def complete(self, prompt: str) -> List[str]:
        """Generate text completions from the prompt."""
        url = f"{self.model_url}/generate"
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": self.temperature,
                "max_length": self.max_tokens,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "num_return_sequences": self.num_return_sequences,
            },
        }
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("generated_text", [])
        except requests.RequestException as e:
            raise Exception(f"Failed to generate text: {e}")

    def chat(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        """Handles chat-based models."""
        formatted_messages = self._to_tgi_messages(messages)
        payload = {
            "messages": formatted_messages,
            "parameters": self._get_all_kwargs(),
        }
        try:
            response = self.client.chat(messages=formatted_messages, **payload)
            tool_calls = response.choices[0].message.tool_calls
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content=response.choices[0].message.content,
                    additional_kwargs={"tool_calls": tool_calls},
                ),
                raw=dict(response),
            )
        except Exception as e:
            raise Exception(f"Chat failed: {e}")

    def _to_tgi_messages(self, messages: Sequence[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert messages to TGI format."""
        return [{"role": msg.role.value, "content": msg.content} for msg in messages]

    def _get_all_kwargs(self) -> Dict[str, Any]:
        """Combine default and additional kwargs."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.additional_kwargs,
        }

    def resolve_tgi_function_call(self):
        """Validate TGI version for function call support."""
        try:
            url = f"{self.model_url}/info"
            model_info = requests.get(url).json()
            tgi_version = model_info.get("version", "0.0.0")
            if version.parse(tgi_version) >= version.parse("2.0.1"):
                return True
            else:
                raise ValueError(f"TGI version incompatible: {tgi_version}")
        except Exception as e:
            raise Exception(f"Error resolving function call support: {e}")

# Example usage
if __name__ == "__main__":
    llm = TextGenerationInference(
        model_url="http://127.0.0.1:8084",
        model_name="openai-community/gpt2",
        temperature=0.7,
        max_tokens=100,
    )
    try:
        result = llm.complete("hare and tortoise story")
        print("Generated Text:", result)
    except Exception as e:
        print("Error:", e)
