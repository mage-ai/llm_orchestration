from typing import List


@transformer
def transform(chunks_arr: List[List[str]], *args, **kwargs):
    nlp = list(kwargs.get('factory_items_mapping').values())[0][0]
    
    tokens = []
    for chunks in chunks_arr:
        arr = []
        for text in chunks:
            doc = nlp(text)
            arr += [token.text for token in doc if not token.is_stop]
        tokens.append(arr)

    return [
        tokens,
    ]