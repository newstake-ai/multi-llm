import os
import logging
from typing import Union

import tiktoken
from openai import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from multi_llm.providers import base_provider
import openai
from openai.types import chat

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OpenAIProvider(base_provider.BaseProvider):
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def get_models(self):
        return {
            "gpt-3.5-turbo-0613": {},
            "gpt-3.5-turbo-16k-0613": {},
            "gpt-3.5-turbo-1106": {"price": 1.0},
            "gpt-3.5-turbo-0125": {"price": 0.5},
            "gpt-4-0314": {"price": 30.0},
            "gpt-4-32k-0314": {"price": 60.0},
            "gpt-4-0613": {"price": 30.0},
            "gpt-4-32k-0613": {"price": 60.0},
            "gpt-4-1106-preview": {"price": 10.0},
            "gpt-4-0125-preview": {"price": 10.0},
        }

    def count_tokens(self, messages, model="gpt-3.5-turbo-0613"):
        """Return the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning("Model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            logger.warning("gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            logger.warning("gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.count_tokens(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            if isinstance(message, chat.ChatCompletionMessage):
                msg = vars(message)
            elif isinstance(message, dict):
                msg = message
            else:
                raise ValueError(f'Unrecognized message: {message}')

            for key, value in msg.items():
                if value is not None:
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name

        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def create_completion(self, messages, model_name, **kwargs) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        completion = self.client.chat.completions.create(messages=messages, model=model_name, **kwargs)
        return completion
