import numpy as np

from mage_ai.data_preparation.models.block.remote.models import RemoteBlock
from mage_ai.settings.repo import get_repo_path


import math
from typing import List, Tuple

import nltk
from gensim import corpora
from gensim.models import LdaMulticore
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


def preprocess_text(text: str) -> Tuple[List[str], List[List[str]]]:
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Tokenize each sentence into words
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

    # Remove stopwords and lemmatize the words
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    preprocessed_sentences = []
    for sentence in tokenized_sentences:
        preprocessed_sentence = [lemmatizer.lemmatize(word.lower()) for word in sentence if word.isalpha() and word.lower() not in stop_words]
        preprocessed_sentences.append(preprocessed_sentence)

    return sentences, preprocessed_sentences


def chunk_text(preprocessed_sentences: List[List[str]]) -> List[List[str]]:
    chunks = []
    for tokens in preprocessed_sentences:
        chunks.append([token.strip() for token in tokens if token.strip()])
    return chunks


buckets = RemoteBlock.load(
    block_uuid='export/mapping/files',
    pipeline_uuid='data_preparation_data_loader',
    repo_path=get_repo_path(),
).get_outputs()

documents = []
for bucket in buckets:
    documents += bucket


sentences, preprocessed_sentences = preprocess_text('\n'.join([document[1] for document in documents]))
num_topics = math.ceil(math.sqrt(len(sentences)))
words_per_topic = max(math.ceil(math.sqrt(num_topics)), 4)

# Chunk the preprocessed text
chunked_text = chunk_text(preprocessed_sentences)

# Create a dictionary from the chunked text
dictionary = corpora.Dictionary(chunked_text)

# Create a corpus from the dictionary
corpus = [dictionary.doc2bow(chunk) for chunk in chunked_text]

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Train the LDA model
lda_model = LdaMulticore(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    # Increase the number of passes over the corpus
    passes=10,
    # Increase the number of iterations per pass
    iterations=400,
)

lda_model