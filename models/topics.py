import math
import os
from typing import List, Tuple

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
    verbose: int = 1,
) -> Tuple[LdaMulticore, corpora.Dictionary, str, str]:
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
        lda_model = LdaMulticore(
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
        lda_model.save(model_file_path)
    else:
        lda_model = LdaMulticore.load(model_file_path)
        dictionary = corpora.Dictionary.load(dictionary_file_path)

    if verbose >= 1:
        print(f'model.num_topics: {lda_model.num_topics}')
        print(f'dictionary.keys: {len(dictionary.keys())}')

    return lda_model, dictionary, model_file_path, dictionary_file_path


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
