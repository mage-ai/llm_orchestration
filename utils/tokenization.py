from typing import Dict, List, Tuple

import numpy as np
import spacy


def named_entity_recognition_tokens(doc: spacy.tokens.doc.Doc) -> Tuple[List[str], List[str]]:
    tokens_text = []
    tokens_type = []

    for token in doc:
        tokens_text.append(token.text)
        tokens_type.append(token.ent_type_ or '<EMPTY>')

    return tokens_text, tokens_type


def standardize(
    doc: spacy.tokens.doc.Doc,
    lemmatize: bool = True,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_stop_words: bool = True,
) -> List[str]:
    tokens = []

    for token in doc:
        tok = token.text.strip()

        if lemmatize:
            tok = token.lemma_

        if lowercase:
            tok = tok.lower()

        if (not remove_punctuation or not token.is_punct) and \
                (not remove_stop_words or not token.is_stop):
            tokens.append(tok)

    return tokens


def word_counts(doc: spacy.tokens.doc.Doc) -> Dict:
    words = {}
    tokens = standardize(doc)

    for token in tokens:
        words[token] = words.get(token, 0)
        words[token] += 1

    return words


def element_wise_addition(
    embeddings_a: List[List[float]],
    embeddings_b: List[List[float]],
) -> List[List[float]]:
    return np.array(embeddings_a) + np.array(embeddings_b)


def element_wise_multiplication(
    embeddings_a: List[List[float]],
    embeddings_b: List[List[float]],
) -> List[List[float]]:
    return np.array(embeddings_a) * np.array(embeddings_b)


def embeddings_concatenate(matrix: List[List[float]]) -> List[float]:
    return np.concatenate(matrix, axis=0)


def embeddings_mean(matrix: List[List[float]]) -> List[float]:
    return np.mean(matrix, axis=0)


def embeddings_sum(matrix: List[List[float]]) -> List[float]:
    return np.sum(matrix, axis=0)


def embeddings_weighted_average(
    matrix: List[List[float]],
    weights: List[float],
) -> List[float]:
    return np.average(matrix, axis=0, weights=weights)


def embeddings_max_pooling(matrix: List[List[float]]) -> List[float]:
    return np.max(matrix, axis=0)
