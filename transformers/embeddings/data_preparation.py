from typing import List, Union

import numpy as np
import torch

import torch
from default_repo.llm_orchestration.utils.tokenization import embeddings_sum
from default_repo.llm_orchestration.utils.tokenization import embeddings_concatenate, embeddings_max_pooling, embeddings_mean


@transformer
def transform(documents: List[List[Union[str, List[str]]]], *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    _, _, _, model = factory_items_mapping['embeddings/clients']

    arr = []
    for document_id, document, metadata, chunk, tokens_pair in documents:
        print(f'document_id: {document_id}')

        # tokens_text, tokens_type = tokens_pair

        # vector_token_text = client.post(tokens_text)
        # vector_token_type = client.post(tokens_text)
        # vector = embeddings_sum([vector_token_text, vector_token_type])
        # Get the subword token ids
        token_ids, attention_mask = tokens_pair

        # Feed the token ids and attention mask to the BERT model
        with torch.no_grad():
            outputs = model(
                torch.from_numpy(np.array(token_ids)), 
                attention_mask=torch.from_numpy(np.array(attention_mask)),
            )

        # Get the embeddings for each subword token
        embeddings = outputs.last_hidden_state

        matrix = embeddings[0].numpy()
        vector = embeddings_concatenate([
            embeddings_mean(matrix),
            embeddings_max_pooling(matrix),
        ])
        
        arr.append([
            document_id,
            document,
            metadata,
            chunk,
            tokens_pair,
            vector,
        ])

        print(f'{round(100 * len(arr) / len(documents))}% ({len(arr)}/{len(documents)})')
    
    return [
        arr,
    ]
