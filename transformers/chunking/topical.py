from typing import List


def chunk_by_text_rank(nlp, text: str):
    doc = nlp(text)
    key_phrases = [span.text for span in doc._.phrases]
    return key_phrases


@transformer
def transform(file_paths: List[str], *args, **kwargs):
    dry_run = 1 == int(kwargs.get('dry_run'))

    nlp = list(kwargs.get('factory_items_mapping').values())[0][0]

    arr = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            text = f.read()
            arr.append(chunk_by_text_rank(nlp, text)[:(100 if dry_run else -1)])
            
    return [arr]