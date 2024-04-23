from typing import List, Tuple

import spacy
from gensim import corpora
from gensim.models import LdaMulticore

from default_repo.llm_orchestration.utils.tokenization import standardize


def chunk_sentences(doc: spacy.tokens.doc.Doc) -> List[str]:
    return [sent.text.strip() for sent in doc.sents]


def chunk_topics(
    nlp: spacy.lang.en.English,
    lda_model: LdaMulticore,
    dictionary: corpora.Dictionary,
    sentences: List[str],
) -> List[List[str]]:
    corpus = []
    for sentence in sentences:
        doc = nlp(sentence)
        tokens = standardize(doc)
        corpus.append(dictionary.doc2bow(tokens))

    topic_distributions = lda_model.get_document_topics(corpus)

    # Segment the text based on topic shifts
    sentences_list = []
    current_sentences = []
    topic_dist_prev = None

    for sentence, topic_dist in zip(sentences, topic_distributions):
        if topic_dist_prev is None or __significant_topic_shift(topic_dist_prev, topic_dist):
            if current_sentences:
                sentences_list.append(current_sentences)
                current_sentences = []

        current_sentences.append(sentence)
        topic_dist_prev = topic_dist

    if current_sentences:
        sentences_list.append(current_sentences)

    return sentences_list


def __significant_topic_shift(topic_dist_prev: List[Tuple], topic_dist: List[Tuple]) -> bool:
    if topic_dist_prev and topic_dist:
        if len(topic_dist_prev) and \
                len(topic_dist_prev[0]) and \
                len(topic_dist) and \
                len(topic_dist[0]):

            return topic_dist_prev[0] != topic_dist[0] and \
                abs(topic_dist_prev[0][1] - topic_dist[0][1]) >= 0.1

    return False
