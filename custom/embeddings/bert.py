from typing import Dict

import numpy as np
import pandas as pd
import torch

from default_repo.llm_orchestration.utils.tokenization import (
    embeddings_concatenate, 
    embeddings_max_pooling, 
    embeddings_mean,
)


@custom
def transform_custom(df: pd.DataFrame, *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    model, _tokenizer = factory_items_mapping['models/bert']

    count = len(df)
    print(f'count: {count}')

    rows = []
    for index, row in df.iterrows():
        token_ids = torch.from_numpy(np.array(row['tokens']))
        attention_mask = torch.from_numpy(np.array(row['attention_mask']))

        # Feed the token ids and attention mask to the BERT model
        with torch.no_grad():
            outputs = model(
                token_ids, 
                attention_mask=attention_mask,
            )

        # Get the embeddings for each subword token
        embeddings = outputs.last_hidden_state

        matrix = embeddings[0].numpy()
        vector = embeddings_concatenate([
            embeddings_mean(matrix),
            embeddings_max_pooling(matrix),
        ])

        row['vector'] = vector.tolist()

        rows.append(row)

        print(f'{index + 1}/{count}')

    return pd.DataFrame(rows)