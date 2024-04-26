import json
import os
import requests
import ssl
from requests.adapters import HTTPAdapter
from typing import Dict, List
# from urllib3 import PoolManager
from urllib3.poolmanager import PoolManager
from urllib3.exceptions import InsecureRequestWarning

import numpy as np
import replicate
from replicate.deployment import DeploymentsPredictions
from transformers import BertModel

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')


def reduce_strings_to_length(strings: List[str], max_combined_length: int) -> List[str]:
    """
    Reduce a list of strings to ensure the combined length of the items does not exceed max_combined_length.
    
    Args:
        strings (List[str]): The list of strings to reduce.
        max_combined_length (int): The maximum combined length of the strings in the result.
    
    Returns:
        List[str]: A list of strings with combined length not exceeding max_combined_length.
    """
    combined_length = 0
    result = []
    
    for string in strings:
        if combined_length + len(string) > max_combined_length:
            break
        result.append(string)
        combined_length += len(string)
    
    return result


class NoSSLVerification(HTTPAdapter):
    """
    An HTTPAdapter that disables SSL certificate verification.
    """
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        # This disables both certificate verification and hostname checking.
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return super().init_poolmanager(*args, **kwargs, ssl_context=context)


class MyAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        # Use SERVER_AUTH for client connections
        pool_kwargs['ssl_context'] = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        pool_kwargs['ssl_context'].options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1  # Disable TLSv1 and TLSv1.1
        pool_kwargs['ssl_context'].set_ciphers('HIGH:!aNULL:!eNULL:!kRSA:!PSK:!SRP:!MD5:!RC4')
        super().init_poolmanager(connections, maxsize, block, **pool_kwargs)


class SessionCohere:
    @classmethod
    def post(self, tokens: List[str]) -> List[float]:
        # Necessary setup to suppress the InsecureRequestWarning if you're doing a verify=False for requests
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

        # Rest of your setup
        api_key = os.getenv('COHERE_API_KEY')
        session = requests.Session()
        session.mount('https://', MyAdapter())

        response = session.post(
            'https://api.cohere.ai/v1/embed',
            data=json.dumps(dict(
                texts=tokens,
                model='embed-english-v3.0',
                input_type='search_document',  # Data preparation
                embedding_types=['float'],
                truncate='END',
                # input_type='search_query',  # Inference
            )),
            headers={
                'accept': 'application/json',
                'content-type': 'application/json',
                'Authorization': f'bearer {api_key}',
            },
        )

        print(response)

        matrix = response.json()['embeddings']['float']
        vector = np.max(matrix, axis=0)

        return vector


class ReplicatePrediction:
    def __init__(self):
        self.prediction = None

    def vector(self) -> List[float]:
        if not self.prediction.get('output'):
            self.get()

        print(self.prediction.get('status'))

        if self.prediction.get('error'):
            raise Exception(self.prediction['error'])

        output = self.prediction.get('output')
        if not output:
            return

        print(f'output: {len(output)}')
        matrix = [r['embedding'] for r in output]

        # Assuming the 3x768 matrix is stored in a variable called 'matrix'
        vector = np.max(matrix, axis=0)
        print(f'shape: {vector.shape}')
        vector = vector.tolist()

        return vector

    def get(self) -> Dict:
        # Necessary setup to suppress the InsecureRequestWarning if you're doing a verify=False for requests
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

        # Rest of your setup
        session = requests.Session()
        session.mount('https://', MyAdapter())

        prediction_id = self.prediction['id']
        response = session.get(
            f'https://api.replicate.com/v1/predictions/{prediction_id}',
            headers={
                'Authorization': f'Bearer {REPLICATE_API_TOKEN}',
                'Content-Type': 'application/json',
            },
        )

        self.prediction = response.json()

        return self.prediction

    def post(self, tokens: List[str]) -> Dict:
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

        session = requests.Session()
        session.mount('https://', MyAdapter())

        response = session.post(
            'https://api.replicate.com/v1/deployments/tommydangerous/mage-dev2/predictions',
            data=json.dumps(dict(input=dict(
                text_batch=json.dumps(tokens),
            ))),
            headers={
                'Authorization': f'Bearer {REPLICATE_API_TOKEN}',
                'Content-Type': 'application/json',
            },
        )

        self.prediction = response.json()

        return self.prediction


class OpenAIEmbedding:
    @classmethod
    def post(self, tokens: List[str]):
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

        session = requests.Session()
        # session.mount("http://", NoSSLVerification())
        # session.mount("https://", NoSSLVerification())
        session.mount('https://', MyAdapter())

        input_data = reduce_strings_to_length(tokens, 8191)[:2048]

        response = session.post(
            'https://api.openai.com/v1/embeddings',
            data=json.dumps(dict(
                input=input_data,
                model='text-embedding-3-large',
                # dimensions=768,
                encoding_format='float',
            )),
            headers={
                'Authorization': f'Bearer {OPENAI_API_KEY}',
                'Content-Type': 'application/json',
            },
            # verify=False,
        )

        res = response.json()
        if not res.get('data'):
            print('[ERROR] OpenAIEmbedding.post')
            print(res)
            print(f'input_data: {input_data}')
            
            raise Exception(json.dumps(res))
            
        return res['data'][0]['embedding']


@factory
def cohere(*args, **kwargs):
    return SessionCohere


@factory
def replicate_prediction(*args, **kwargs):
    return ReplicatePrediction


@factory
def openai(*args, **kwargs):
    return OpenAIEmbedding


@factory
def bert_model(*args, **kwargs):
    return BertModel.from_pretrained('bert-base-uncased')
