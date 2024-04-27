from typing import List, Union

import numpy as np


def max_pool_vectors(vectors):
    # Stack the vectors along a new axis to create a 2D array
    stacked_vectors = np.stack(vectors, axis=0)
    
    # Perform max pooling along the first axis (axis=0)
    max_pooled_vector = np.max(stacked_vectors, axis=0)
    
    return max_pooled_vector


@transformer
def transform(documents: List[List[Union[str, List[str]]]], *args, **kwargs):
    max_dimensions = kwargs.get('max_dimensions', 1536)
    
    print(f'documents: {len(documents)}')
    documents_more = []
    for document_id, document, chunk, tokens, embeddings_list in documents:
        print(f'document_id: {document_id}')

        documents_more.append([
            document_id,
            document,
            chunk,
            tokens,
            max_pool_vectors(embeddings_list).tolist(),
        ])

        print(f'{round(100 * len(documents_more) / len(documents))}% ({len(documents_more)}/{len(documents)})')
    
    return [
        documents_more,
    ]