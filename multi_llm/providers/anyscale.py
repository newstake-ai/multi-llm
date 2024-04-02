import os
from multi_llm.providers import openai


class AnyscaleProvider(openai.OpenAIProvider):
    def __init__(self):
        self.client = openai.openai.OpenAI(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key=os.getenv("ANYSCALE_API_KEY"),
        )

    def get_models(self):
        return {
            "google/gemma-7b-it": {"price": 0.15},
            "meta-llama/Llama-2-7b-chat-hf": {"price": 0.15},
            "mistralai/Mistral-7B-Instruct-v0.1": {"price": 0.15},
            "mlabonne/NeuralHermes-2.5-Mistral-7B": {"price": 0.15},
            "meta-llama/Llama-2-13b-chat-hf": {"price": 0.25},
            "mistralai/Mixtral-8x7B-Instruct-v0.1": {"price": 0.5},
            "meta-llama/Llama-2-70b-chat-hf": {"price": 1.0},
            "codellama/CodeLlama-70b-Instruct-hf": {"price": 1.0},
        }