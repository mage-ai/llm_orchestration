import requests
from enum import Enum
from typing import Callable, Dict, List


class Transformers:
    class Model(str, Enum):
        BERT = 'bert-large-uncased'
        CODE_LLAMA = 'Phind/Phind-CodeLlama-34B-v2'
        DEEPSEEKER_BASE = 'deepseek-ai/deepseek-coder-6.7b-base'
        DEEPSEEKER_INSTRUCT = 'deepseek-ai/deepseek-coder-6.7b-instruct'
        INSTRUCTOR = 'hkunlp/instructor-xl'
        OPEN_CODE_INTERPRETER = 'm-a-p/OpenCodeInterpreter-DS-33B'

    def __init__(cls, model: Model = Model.MODEL_BERT):
        cls.model = model

    def create_embeddings(self, texts: List[str], **kwargs) -> Dict:
        return self.__post('embeddings', texts=texts, **kwargs)

    def create_generations(self, texts: List[str], **kwargs) -> Dict:
        return self.__post('generations', texts=texts, **kwargs)

    def create_messages(self, messages: List[Dict], **kwargs) -> Dict:
        """
        messages: List[Dict]
            {
                "role": "user|assistant",
                "content": "hello world"
            }
        """
        return self.__post('messages', messages=messages, **kwargs)

    def __post(self, resource: str, **kwargs) -> Dict:
        payload = dict(model=self.model)
        payload.update(kwargs or {})

        response = requests.post(f'https://sorcery.mage.ai/{resource}', json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f'Failed to fetch {resource}: {response.text}')


@factory
def bert(*args, **kwargs) -> Transformers
    return Transformers(Transformers.Model.BERT)


@factory
def code_llama(*args, **kwargs) -> Transformers
    return Transformers(Transformers.Model.CODE_LLAMA)


@factory
def deepseeker_base(*args, **kwargs) -> Transformers
    return Transformers(Transformers.Model.DEEPSEEKER_BASE)


@factory
def deepseeker_instruct(*args, **kwargs) -> Transformers
    return Transformers(Transformers.Model.DEEPSEEKER_INSTRUCT)


@factory
def instructor(*args, **kwargs) -> Transformers
    return Transformers(Transformers.Model.INSTRUCTOR)


@factory
def open_code_interpreter(*args, **kwargs) -> Transformers
    return Transformers(Transformers.Model.OPEN_CODE_INTERPRETER)
