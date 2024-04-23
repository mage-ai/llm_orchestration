import math
import os
from typing import Dict, List, Tuple

import spacy
from gensim import corpora
from gensim.models import LdaMulticore

from mage_ai.settings.repo import get_repo_path

from default_repo.llm_orchestration.utils.chunking import chunk_sentences
from default_repo.llm_orchestration.utils.tokenization import standardize


def get_train_transform(
    nlp: spacy.lang.en.English,
    documents: List[str] = None,
    execution_partition: str = None,
    random_state: int = 3,
    train: bool = False,
    transform_text: str = None,
    verbose: int = 1,
) -> Dict:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    model_file_path, dictionary_file_path = __setup_files(execution_partition=execution_partition)

    if train:
        documents_count = len(documents) if documents else None
        if verbose >= 1:
            print(f'Training on {documents_count} documents.')

        num_topics = round(math.sqrt(documents_count) * 10)
        if num_topics >= documents_count:
            num_topics = round(documents_count / 2)
        num_topics = max(num_topics, 1)

        chunked_tokens = []
        for text in documents:
            sentences = chunk_sentences(nlp(text))
            if verbose >= 1:
                print(f'sentences: {len(sentences)}')

            for sentence in sentences:
                tokens = standardize(nlp(sentence))
                chunked_tokens.append(tokens)

        if verbose >= 1:
            print(f'chunked_tokens: {len(chunked_tokens)}')

        # Create a dictionary from the chunked text
        dictionary = corpora.Dictionary(chunked_tokens)
        dictionary.save(dictionary_file_path)

        # Create a corpus from the dictionary
        corpus = [dictionary.doc2bow(tokens) for tokens in chunked_tokens]

        # Train the LDA model
        model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            # Increase the number of iterations per pass
            iterations=400,
            num_topics=num_topics,
            # Increase the number of passes over the corpus
            passes=10,
            random_state=random_state,
        )

        if verbose >= 1:
            print('Saving model...')
        model.save(model_file_path)
    else:
        model = LdaMulticore.load(model_file_path)
        dictionary = corpora.Dictionary.load(dictionary_file_path)

    if verbose >= 1:
        print(f'model.num_topics: {model.num_topics}')
        print(f'dictionary.keys: {len(dictionary.keys())}')

    tokens = None
    if transform_text:
        tokens = __chunk_topics(nlp, transform_text, dictionary=dictionary, model=model)

    return dict(
        dictionary=dictionary,
        model=model,
        paths=[model_file_path, dictionary_file_path],
        tokens=tokens,
    )


def extract_topics(model: LdaMulticore, num_words: int = 8):
    topics = model.show_topics(num_topics=model.num_topics, num_words=num_words, formatted=False)
    for topic_num, topic in topics:
        print(f"Topic: {topic_num}")
        for word, probability in topic:
            print(f"    {word}: {probability}")


def __chunk_topics(
    nlp: spacy.lang.en.English,
    text: str,
    dictionary: corpora.Dictionary = None,
    model: LdaMulticore = None,
) -> List[List[str]]:
    corpus = []
    sentences = chunk_sentences(nlp(text))
    for sentence in sentences:
        tokens = standardize(nlp(sentence))
        corpus.append(dictionary.doc2bow(tokens))

    topic_distributions = model.get_document_topics(corpus)

    # Segment the text based on topic shifts
    groups = []
    current = []
    topic_dist_prev = None

    for sentence, topic_dist in zip(sentences, topic_distributions):
        if topic_dist_prev is None or __significant_topic_shift(topic_dist_prev, topic_dist):
            if current:
                groups.append(current)
                current = []

        current.append(sentence)
        topic_dist_prev = topic_dist

    if current:
        groups.append(current)

    return groups


def __significant_topic_shift(topic_dist_prev: List[Tuple], topic_dist: List[Tuple]) -> bool:
    if topic_dist_prev and topic_dist:
        if len(topic_dist_prev) and \
                len(topic_dist_prev[0]) and \
                len(topic_dist) and \
                len(topic_dist[0]):

            return topic_dist_prev[0] != topic_dist[0] and \
                abs(topic_dist_prev[0][1] - topic_dist[0][1]) >= 0.1

    return False


def __setup_files(execution_partition: str = None) -> Tuple[str, str]:
    model_dir = os.path.join(get_repo_path(), 'assets', 'models')
    dictionary_dir = os.path.join(get_repo_path(), 'assets', 'dictionary')

    if not execution_partition:
        model_dir = os.path.join(model_dir, 'notebook')
        dictionary_dir = os.path.join(dictionary_dir, 'notebook')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(dictionary_dir, exist_ok=True)

    model_file_path = os.path.join(model_dir, 'lda.model')
    dictionary_file_path = os.path.join(dictionary_dir, 'lda')

    return model_file_path, dictionary_file_path
