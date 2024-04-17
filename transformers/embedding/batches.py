import os
from typing import List

import voyageai
# from chromadb.utils import embedding_functions
# from sentence_transformers import SentenceTransformer


# def batch_embeddings(tokens: List[str]) -> List[int]:
#     client = voyageai.Client(api_key=os.getenv('VOYAGEAI_API_KEY'))
#     model = 'voyage-large-2'
#     embeddings_result = client.embed(tokens[:128], model=model)
    
#     return embeddings_result.embeddings


@transformer
def transform(tokens_for_chunks_for_documents: List[List[str]], *args, **kwargs):
    """
    Index level 0: single document and its chunks (List[List[str]])
        Index level 1: single chunk and its tokens (List[str])
            Index level 2: token (str)
    """
    # default_ef = embedding_functions.DefaultEmbeddingFunction()

    model = list(kwargs.get('factory_items_mapping').values())[0][0]
    
    embeddings_for_tokens_for_chunks_for_documents = []
    for idx1, pair in enumerate(tokens_for_chunks_for_documents):
        print('tokens_for_chunks', idx1)

        file_path, tokens_for_chunks = pair

        embeddings_for_tokens_for_chunks = []
        for idx2, tokens_for_chunk in enumerate(tokens_for_chunks):
            # Fastest one
            embeddings_for_tokens_for_chunk = model.encode(tokens_for_chunk)[0]
            # embeddings_for_tokens_for_chunk = default_ef(tokens_for_chunk)
            # embeddings_for_tokens_for_chunk = batch_embeddings(tokens_for_chunk)
            embeddings_for_tokens_for_chunks.append(embeddings_for_tokens_for_chunk)
            
        embeddings_for_tokens_for_chunks_for_documents.append([file_path, embeddings_for_tokens_for_chunks])

    """
    [ tokens_for_chunks_for_documents
        [ tokens_for_chunks
            [ tokens_for_chunk
                [ embeddings_for_token  
                    1,
                    2,
                    3,
                ],
            ],
            [],
        ],
        [],
        [], 
    ]
    """

    return [
        embeddings_for_tokens_for_chunks_for_documents,
    ]