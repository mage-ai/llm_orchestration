import os
from typing import Dict, List, Tuple, Union

import sentencepiece as spm
import spacy

from mage_ai.settings.repo import get_repo_path
from default_repo.llm_orchestration.utils.tokenization import word_counts


def get_train_transform(
    nlp: spacy.lang.en.English,
    documents: List[str] = None,
    execution_partition: str = None,
    model_type: str = 'unigram',
    train: bool = False,
    transform_text: str = False,
    verbose: int = 1,
    vocab_size: str = None,
) -> Tuple[spm.SentencePieceProcessor, str, Union[None, List[str]]]:
    """
    model_type:
        bpe
        unigram
        word_piece
    """

    model_file_path = __setup_files(execution_partition=execution_partition)

    if train:
        doc = nlp('\n'.join(documents))
        docs_count = len(documents)
        word_count = len(word_counts(doc))

        if not vocab_size:
            vocab_size = round(min(
                (word_count/docs_count) * (docs_count**0.5),
                (word_count/docs_count),
            ))

        if verbose >= 1:
            print(
                f'Training {docs_count} documents '
                f'with {word_count} words and '
                f'vocabulary size {vocab_size}.',
            )

        model = spm.SentencePieceTrainer.train(
            model_prefix=model_file_path,
            model_type=model_type,
            sentence_iterator=iter(documents),
            vocab_size=vocab_size,
        )
    else:
        model = spm.SentencePieceProcessor()
        model.load(f'{model_file_path}.model')

    tokens = None
    if transform_text:
        tokens = model.encode_as_pieces(transform_text)

    return model, model_file_path, tokens


def __setup_files(execution_partition: str = None) -> str:
    model_dir = os.path.join(get_repo_path(), 'assets', 'models')

    if not execution_partition:
        model_dir = os.path.join(model_dir, 'notebook')

    os.makedirs(model_dir, exist_ok=True)

    model_file_path = os.path.join(model_dir, 'subword_tokenizer')

    return model_file_path
