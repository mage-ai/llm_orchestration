import os
from typing import List

import voyageai  # Assumed to be a fictional or third-party library for demonstration

MAX_TEXTS = 128
MAX_TOKENS = 120_000

def batch_embeddings(
    tokenized_chunks: List[List[str]],
    verbosity: int = 1,
) -> List[List[List[int]]]:
    client = voyageai.Client(api_key=os.getenv('VOYAGEAI_API_KEY'))
    model = 'voyage-large-2'

    embeddings_grouped = []  # This will store embeddings for each chunk.
    api_calls = 0  # To keep track of API calls made.

    for chunk in tokenized_chunks:
        # Count the tokens in this chunk to check against MAX_TOKENS;
        # adjust logic if handling of large chunks is required.
        chunk_token_count = sum(len(token) for token in chunk)
        
        # Check if this chunk exceeds MAX_TOKENS; if it does, you might split it,
        # but for this example, we'll process it as is, assuming it fits.
        if chunk_token_count > MAX_TOKENS:
            # Handle large chunks appropriately here.
            print(f"Warning: Chunk with {chunk_token_count} tokens exceeds the limit and needs special handling.")
            continue  # Skip this chunk or split it as needed. 
        
        embeddings_result = client.embed(chunk, model=model)
        api_calls += 1
        if verbosity >= 1:
            print(f'API calls {api_calls}, processed chunk with {chunk_token_count} tokens')

        # Assuming `.embed` returns embeddings in the same order as texts,
        # we can directly append the result for this chunk.
        embeddings_grouped.append(embeddings_result.embeddings)

    return embeddings_grouped