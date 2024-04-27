import os
from typing import List
from uuid import uuid4

import pandas as pd
import voyageai


def get_embeddings_in_batches(sentences_with_structure: List[dict], batch_size: int, model: str, client, verbosity: int) -> List:
    all_embeddings = []
    total_batches = (len(sentences_with_structure) + batch_size - 1) // batch_size
    if verbosity >= 1:
        print(f"Total sentences: {len(sentences_with_structure)} | Total batches to process: {total_batches}")
    for i in range(0, len(sentences_with_structure), batch_size):
        batch = sentences_with_structure[i:i+batch_size]
        batch_sentences = [sentence['text'] for sentence in batch]
        batch_ids = [sentence['id'] for sentence in batch]
        if verbosity >= 2:
            print(f"Processing batch {i // batch_size + 1} of {total_batches}")
        try:
            embeddings_result = client.embed(batch_sentences, model=model)  # Assuming this returns embeddings in order
            batch_embeddings = embeddings_result.embeddings
            for idx, emb in enumerate(batch_embeddings):
                all_embeddings.append({
                    "sentence_id": batch_ids[idx],
                    "vector": emb
                })
            if verbosity >= 2:
                print(f"Batch {i // batch_size + 1} processed successfully.")
        except Exception as e:
            if verbosity >= 1:
                print(f"Error in batch processing: {e}")
            all_embeddings.extend([{"sentence_id": id, "vector": None} for id in batch_ids])
    return all_embeddings


def create_documents_from_texts(texts: List[dict], verbosity: int):
    documents = []
    vo = voyageai.Client(api_key='Your-API-Key')
    batch_size = 128
    if verbosity >= 1:
        print(f"Total documents to process: {len(texts)}")
    
    for idx, document in enumerate(texts):
        text = document.get('text', '')
        sentences_texts = document.get('sentences', [])
        sentences_with_structure = [{"id": str(uuid4()), "text": sentence} for sentence in sentences_texts]
        if sentences_with_structure:
            embeddings = get_embeddings_in_batches(sentences_with_structure, batch_size, "voyage-large-2", vo, verbosity)
            for sentence in sentences_with_structure:
                matching_embedding = next((emb for emb in embeddings if emb['sentence_id'] == sentence['id']), None)
                if matching_embedding:
                    sentence['embeddings'] = matching_embedding['vector']
        
        processed_document = {
            'id': str(uuid4()),  # Assuming you still want a unique identifier for each document
            'text': text,
            'sentences': sentences_with_structure,  # Structured sentences with embeddings
        }
        documents.append(processed_document)
        if verbosity >= 1:
            print(f"Document {idx+1} processed.")
    return documents


@transformer
def load_data(full_file_paths, *args, **kwargs):
    verbosity = kwargs.get('verbosity', 1)
    documents = create_documents_from_file_paths(full_file_paths, verbosity)
    if verbosity >= 1:
        print("Data loading complete.")
    if verbosity >= 2:
        print(documents)

    return [documents]
