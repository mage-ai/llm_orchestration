import json
import os
import requests
import ssl
from requests.adapters import HTTPAdapter
from typing import Dict, List
from urllib3.poolmanager import PoolManager
from urllib3.exceptions import InsecureRequestWarning

import numpy as np

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


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


class OpenAIEmbedding:
    @classmethod
    def post(cls, tokens: List[str]):
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


class OpenAIChatCompletion:
    @classmethod
    def post(cls, messages: List[Dict[str, str]], model='gpt-4-turbo'):
        """
        Sends messages to the OpenAI chat API and receives a response.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries with 'role' and 'content' keys.
            model (str, optional): Model to be used for the chat completion.
            response_format (dict, optional): The desired format of the response from the API.

        Returns:
            str: Content of the response from the OpenAI chat completion API.
        """

        # Disable SSL warnings
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        session = requests.Session()
        # Consider applying custom or no SSL verification as in OpenAIEmbedding
        session.mount('https://', MyAdapter())

        payload = json.dumps({
            "model": model,
            "messages": messages,
        })

        response = session.post(
            'https://api.openai.com/v1/chat/completions',
            data=payload,
            headers={
                'Authorization': f'Bearer {OPENAI_API_KEY}',
                'Content-Type': 'application/json',
            },
        )

        if response.status_code == 200:
            return response.json()
        else:
            print('[ERROR] OpenAIChatCompletion.post')
            print(response.text)
            raise Exception(f"API request failed with status code {response.status_code}")
