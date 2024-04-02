from typing import Dict
import abc

from openai.types.chat import ChatCompletion


class BaseProvider(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_models(self) -> Dict[str, Dict[str, str]]:
        pass

    @abc.abstractmethod
    def count_tokens(self, messages, model_name) -> int:
        pass

    @abc.abstractmethod
    def create_completion(self, messages, model_name, **kwargs) -> ChatCompletion:
        pass