from typing import List

import torch
from transformers import AutoModel, AutoTokenizer


def semantic_chunker(model: AutoModel, tokenizer: AutoTokenizer, text: str, chunk_size: int = 512) -> List[str]:
    """
    Split text into chunks based on semantic similarity.

    Args:
        model (AutoModel): The transformer model used for embedding.
        tokenizer (AutoTokenizer): The tokenizer used for encoding text.
        text (str): The input text to be chunked.
        chunk_size (int): The desired size of each chunk in tokens, default to 512.

    Returns:
        List[str]: A list of text chunks.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Move model to the appropriate device.
    model = model.to(device)

    # Encode the text to tokens and truncate/pad to chunk_size.
    # Using `max_length` to ensure we do not exceed model limitations.
    tokens = tokenizer.encode_plus(
        text, 
        return_tensors='pt', 
        max_length=chunk_size, 
        truncation=True, 
        padding='max_length',
    )

    # Moving encoded tokens to the same device as model
    tokens_tensor = tokens['input_ids'].to(device)
    attention_mask_tensor = tokens['attention_mask'].to(device)

    # Process chunks
    chunks = []
    for i in range(0, tokens_tensor.size(1), chunk_size):
        # Process each chunk through the model
        chunk_tokens_tensor = tokens_tensor[:, i:i + chunk_size]
        chunk_attention_mask_tensor = attention_mask_tensor[:, i:i + chunk_size]

        with torch.no_grad():
            # Getting the last hidden state
            chunk_embeddings = model(
                input_ids=chunk_tokens_tensor, 
                attention_mask=chunk_attention_mask_tensor,
            )[0]

        chunk_text = tokenizer.decode(chunk_tokens_tensor[0], skip_special_tokens=True)
        chunks.append(chunk_text)

    return chunks