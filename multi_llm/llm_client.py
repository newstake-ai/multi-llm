import logging
from typing import List

from openai.types.chat import ChatCompletion
from multi_llm import providers


class MultiLLMClient(object):
    def __init__(self, providers: List[providers.BaseProvider]):
        self.providers = providers

    def get_models(self):
        all_models = []
        for provider in self.providers:
            all_models.extend(provider.get_models())

    def count_tokens(self, messages, model_name):
        provider, _ = self._select_provider(model_name)
        return provider.count_tokens(messages, model_name)

    def create_completion(self, messages, model_name, **kwargs) -> ChatCompletion:
        provider, fallback = self._select_provider(model_name)
        try:
            return provider.create_completion(messages, model_name, **kwargs)
        except Exception as e:
            logging.warning(f'Encountered error with {provider}: {e}\nFalling back to {fallback}...')
            return fallback.create_completion(messages, model_name, **kwargs)

    def _select_provider(self, model_name):
        selected_provider = None
        fallback_provider = None
        for provider in self.providers:
            models_and_prices = provider.get_models()
            if model_name in models_and_prices.keys() and selected_provider is None:
                selected_provider = provider
                fallback_provider = provider
            elif model_name in models_and_prices.keys():
                if models_and_prices[model_name].get('price') < selected_provider.get_models()[model_name].get('price'):
                    fallback_provider = selected_provider
                    selected_provider = provider
        if selected_provider is None:
            raise ValueError(f'No provider found for {model_name}!')
        return selected_provider, fallback_provider
