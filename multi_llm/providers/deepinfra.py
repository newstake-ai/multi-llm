import os

from multi_llm.providers import openai


class DeepinfraProvider(openai.OpenAIProvider):
    def __init__(self):
        self.client = openai.openai.OpenAI(
            base_url="https://api.deepinfra.com/v1/openai",
            api_key=os.getenv("DEEPINFRA_TOKEN"),
        )

    def get_models(self):
        return {
            "meta-llama/Llama-2-7b-chat-hf": {"price": 0.13},
            "mistralai/Mistral-7B-Instruct-v0.1": {"price": 0.13},
            "openchat/openchat_3.5": {"price": 0.13},
            "meta-llama/Llama-2-13b-chat-hf": {"price": 0.22},
            "Gryphe/MythoMax-L2-13b": {"price": 0.22},
            "mistralai/Mixtral-8x7B-Instruct-v0.1": {"price": 0.27},
            "01-ai/Yi-34B-Chat": {"price": 0.60},
            "codellama/CodeLlama-34b-Instruct-hf": {"price": 0.60},
            "Phind/Phind-CodeLlama-34B-v2": {"price": 0.60},
            "meta-llama/Llama-2-70b-chat-hf": {"price": 0.70},
            "deepinfra/airoboros-70b": {"price": 0.70},
            "lizpreciatior/lzlv_70b_fp16_hf": {"price": 0.70},
        }