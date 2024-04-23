from typing import Dict, List

import spacy


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
