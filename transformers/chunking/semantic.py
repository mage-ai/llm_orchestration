import math
from typing import List, Tuple

import nltk
from gensim import corpora
from gensim.models import LdaMulticore
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

from default_repo.llm_orchestration.transformers.topics import chunk_sentences, group_sentences

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
    

def significant_topic_shift(prev_topic_dist: List[Tuple], topic_dist: List[Tuple]) -> bool:
    if prev_topic_dist and topic_dist:
        if len(prev_topic_dist) and \
                len(prev_topic_dist[0]) and \
                len(topic_dist) and \
                len(topic_dist[0]):

            return prev_topic_dist[0] != topic_dist[0] and \
                abs(prev_topic_dist[0][1] - topic_dist[0][1]) >= 0.1
    return False


def chunk_by_topic(
    nlp,
    model_file_path: str,
    dictionary_file_path: str,
    stop_words,
    document: str,
) -> List[str]:
    lda_model = LdaMulticore.load(model_file_path)
    dictionary = corpora.Dictionary.load(dictionary_file_path)

    print(f'model.num_topics: {lda_model.num_topics}')
    print(f'dictionary.keys: {len(dictionary.keys())}')

    sentences = chunk_sentences(nlp, document)
    print(f'sentences: {len(sentences)}')

    if not sentences:
        return []

    corpus = []
    for sentence in sentences:
        doc = nlp(sentence)
        corpus.append(dictionary.doc2bow(
            [token.lemma_ for token in doc if not token.is_punct],        
        ))
    topic_distributions = lda_model.get_document_topics(corpus)

    print(f'topic_distributions: {len(topic_distributions)}')

    # Segment the text based on topic shifts
    segments = []
    current_segment = []
    prev_topic_dist = None

    for sentence, topic_dist in zip(sentences, topic_distributions):
        print(f'sentence: {len(sentence)}')
        print(f'topic_dist: {topic_dist}')
        if prev_topic_dist is None or significant_topic_shift(prev_topic_dist, topic_dist):
            if current_segment:
                print('Topic changed...')
                segments.append(' '.join(current_segment))
                current_segment = []
        current_segment.append(sentence)
        prev_topic_dist = topic_dist

    if current_segment:
        segments.append(' '.join(current_segment))

    print(f'segments: {len(segments)}')

    return segments
    
    # Generate topic representations for each chunk
    # topic_representations = []
    # for idx, chunk in enumerate(chunked_text):
    #     chunk_bow = dictionary.doc2bow(chunk)
    #     topic_distribution = lda_model[chunk_bow]
    #     if topic_distribution:
    #         topic_representations.append(topic_distribution)
    #     else:
    #         print(f'No topic distribution for chunk at index {idx}: {chunk}')

    # topics_mapping = {}
    # for topic in lda_model.show_topics(num_topics=lda_model.num_topics):
    #     topic_id = topic[0]

    #     words = []
    #     for word in topic[1].split('+'):
    #         parts = word.split('*')
    #         part = parts[1] if len(parts) >= 2 else parts[0]
    #         words.append(part.strip().strip('"'))

    #     topics_mapping[topic_id] = dict(
    #         chunks=[],
    #         words=words,
    #     )

    # arr = []
    # for idx, representation in enumerate(topic_representations):
    #     topic_id, probability = sorted(representation, key=lambda t: t[1], reverse=True)[0]
    #     chunks = chunked_text[idx]
    #     for chunk in chunks:
    #         arr.append((topic_id, chunk))

    # return arr


def sliding_window_chunker(text: str, chunk_size: int = 256, chunk_overlap: int = 64):
    """
    Split text into chunks using a sliding window approach.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The desired size of each chunk in tokens.
        chunk_overlap (int): The number of overlapping tokens between consecutive chunks.

    Returns:
        list: A list of text chunks.
    """
    tokens = nltk.word_tokenize(text)
    chunks = []
    start = 0
    end = chunk_size

    while start < len(tokens):
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens)
        chunks.append(chunk_text)
        start = end - chunk_overlap
        end = start + chunk_size

    return chunks


@transformer
def transform(documents: List[List[str]], file_paths: List[str], *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    nlp, stop_words = factory_items_mapping['data_preparation/nlp']
    model_file_path, dictionary_file_path = file_paths

    arr = []
    for document_id, document, metadata in documents:
        paragraphs = chunk_by_topic(
            nlp,
            model_file_path,
            dictionary_file_path,
            stop_words,
            document,
        )

        print(f'document_id {document_id}')
        print(f'paragraphs: {len(paragraphs)}')

        counter = 0
        for paragraph in paragraphs:
            chunks = sliding_window_chunker(paragraph)
            print(f'chunks: {len(chunks)}')
            
            for chunk in chunks:
                arr.append([
                    document_id,
                    document,
                    metadata,
                    chunk,
                ])

                counter += 1
                print(f'{counter}/{len(chunks)}')

    return [
        arr,
    ]
