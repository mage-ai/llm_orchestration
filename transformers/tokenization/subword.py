import string
from typing import List

import numpy as np
import sentencepiece as spm
from nltk.tokenize import TreebankWordTokenizer

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


def preprocessing(nlp, chunk: str) -> List[str]:
    doc = nlp(chunk)
    tokens = [tok.text for tok in doc if not tok.is_punct]
    text = ' '.join(tokens)

    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop]
    text = ' '.join(tokens)

    doc = nlp(text)
    tokens = [token.lemma_ for token in doc]

    return tokens


def part_of_speech(nlp, chunk: str):
    doc = nlp(chunk)

    tokens = []
    for token in doc:
        token_text = token.text.strip()
        if not token_text:
            continue
        
        token_type = token.pos_
        if token_type:
            tokens.append(f'<{token_type}>{token_text}</{token_type}>')
        else:
            tokens.append(token_text)

    return tokens


def named_entity_recognition(nlp, chunk: str) -> List[str]:
    doc = nlp(chunk)

    tokens = []
    for token in doc:
        token_text = token.text.strip()
        if not token_text:
            continue

        entity_type = token.ent_type_
        if entity_type:
            tokens.append(f'<{entity_type}>{token_text}</{entity_type}>')
        else:
            tokens.append(token_text)

    return tokens


def subword_tokens(model_file_path: str, chunk: str) -> List[str]:
    # Load the trained model
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_file_path}.model')

    tokenized_sentence = sp.encode_as_pieces(chunk)

    # Decode the tokenized sentence back to the original text
    # decoded_sentence = sp.decode_pieces(tokenized_sentence)
    # print("Decoded sentence:", decoded_sentence)
    
    # Tokenize a technical sentence
    return tokenized_sentence


def combo_tokens(nlp, model_file_path: str, stop_words, chunk: str) -> List[str]:
    # Tree bank
    # tokenizer = TreebankWordTokenizer()
    # tokens_treebank = tokenizer.tokenize(chunk)
    # tokens_treebank = [token for token in tokens_treebank if token.lower() not in stop_words]
    # tokens_treebank = [token for token in tokens_treebank if token not in string.punctuation]

    # Named entity recognition
    # tokens_ner = preprocessing(nlp, chunk)
    # tokens_ner = named_entity_recognition(nlp, ' '.join(tokens_ner))

    # Part of speech
    # tokens_pos = preprocessing(nlp, chunk)
    # tokens_pos = part_of_speech(nlp, ' '.join(tokens_pos))

    # Subword
    tokens_subword = subword_tokens(model_file_path, chunk)
    tokens_subword = [token for token in tokens_subword if token.lower() not in stop_words]
    tokens_subword = [token for token in tokens_subword if token not in string.punctuation]

    # Order of tokens matter when we truncate during the encoding/embedding phase
    tokens = np.concatenate([
        # tokens_treebank,
        # tokens_ner,
        # tokens_pos,
        tokens_subword,
    ])

    print(', '.join([
        f'chunk: {len(chunk)}',
        # f'treebank: {len(tokens_treebank)}',
        # f'ner: {len(tokens_ner)}',
        # f'pos: {len(tokens_pos)}',
        # f'subword: {len(tokens_subword)}',
        f'tokens: {len(tokens)}',
    ]))
    
    return tokens


@transformer
def transform(documents: List[List[str]], model_file_path: str, *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    nlp, stop_words = factory_items_mapping['data_preparation/nlp']

    documents_more = []
    
    for document_id, document, metadata, chunk in documents:
        tokens = combo_tokens(nlp, model_file_path, stop_words, chunk)

        documents_more.append([
            document_id,
            document,
            metadata,
            chunk,
            tokens,
        ])

    return [
        documents_more,
    ]
