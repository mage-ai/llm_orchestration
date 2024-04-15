from typing import List

from default_repo.llm_orchestration.utils.chunking import semantic_chunker


@transformer
def transform(file_paths: List[str], *args, **kwargs):
    arr = list(kwargs.get('factory_items_mapping', {}).values())[0]
    model = arr[0]
    tokenizer = arr[1]

    processed_texts = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            text = f.read()
            chunks = semantic_chunker(model, tokenizer, text, chunk_size=kwargs.get('chunk_size', 512))
            processed_texts.append(chunks)

    return [processed_texts]