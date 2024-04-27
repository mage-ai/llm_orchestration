from typing import Dict, List

import pandas as pd


@custom
def transform_custom(data: List[Dict], *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    _model, tokenizer = factory_items_mapping['models/bert']

    rows = []
    for row in data:
        chunk = row['chunk']
        encoded = tokenizer.encode_plus(chunk, return_tensors='pt')
        token_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        row['tokens'] = token_ids.numpy().tolist()
        row['attention_mask'] = attention_mask.numpy().tolist()
        
        rows.append(row)

    return pd.DataFrame(rows)