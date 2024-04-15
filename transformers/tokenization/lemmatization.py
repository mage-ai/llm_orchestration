from typing import List


@transformer
def transform(chunks_arr: List[List[str]], *args, **kwargs):
    nlp = list(kwargs.get('factory_items_mapping').values())[0][0]
    
    tokens = []
    for chunks in chunks_arr:
        lemmatized_tokens = []
        for text in chunks:
            doc = nlp(text)
            lemmatized_tokens += [token.lemma_ for token in doc]
        tokens.append(lemmatized_tokens)

    return [
        tokens,
    ]