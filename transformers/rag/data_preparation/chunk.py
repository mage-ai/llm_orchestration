from typing import List

import spacy


nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # Disabling Named Entity Recognition and Parser to speed up processing
nlp.add_pipe('sentencizer')  # Adding sentencizer to the pipeline, parser is not needed for sentence segmentation


def split_into_sentences_spacy(text: str) -> List[str]:
    doc = nlp(text)
    return [sentence.text.strip() for sentence in doc.sents]


def prepare_texts_for_processing(texts: List[str]) -> List[dict]:
    """
    Given a list of texts, returns a list of dictionaries with each dictionary
    containing the original text and its split sentences.

    :param texts: List of raw text strings to be processed.
    :return: List of dictionaries with 'text' and 'sentences' keys.
    """
    processed_texts = []
    for text in texts:
        sentences = split_into_sentences_spacy(text)
        processed_text = {
            'text': text,
            'sentences': sentences
        }
        processed_texts.append(processed_text)
    return processed_texts


def read_texts_from_file_paths(file_paths: List[str]) -> List[str]:
    """
    Given a list of file paths, reads the content of each file and returns a list of texts.

    :param file_paths: List of strings, where each string is a file path to be read.
    :return: List of strings, where each string is the content of a file.
    """
    texts = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                texts.append(content)
        except IOError as e:
            print(f"Error reading file {file_path}: {e}")

    return texts


def chunk_text(text: str):
    """
    Extract noun phrases from the given text as chunks.
    """
    doc = nlp(text)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    return noun_phrases


@transformer
def load_data(file_paths, *args, **kwargs):
    texts = read_texts_from_file_paths(file_paths)
    prepared_texts = prepare_texts_for_processing(texts)

    return prepared_texts