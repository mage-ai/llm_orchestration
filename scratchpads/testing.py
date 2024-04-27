import numpy as np

import nltk
import pytextrank
import spacy
import split_markdown4gpt
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('textrank')

from mage_ai.data_preparation.models.block.remote.models import RemoteBlock
from mage_ai.settings.repo import get_repo_path

from default_repo.llm_orchestration.utils.chunking import chunk_sentences
from default_repo.llm_orchestration.utils.tokenization import named_entity_recognition_tokens


outputs = RemoteBlock.load(
    block_uuid='export/mapping/files',
    pipeline_uuid='data_preparation_data_loader',
    repo_path=get_repo_path(),
).get_outputs()


documents0 = []
for output in outputs:
    for arr in output:
        documents0.append(arr)

print(len(documents0))
    
tokenizer = TreebankWordTokenizer()

documents = []
for document_id, document, metadata in documents0:
    chunks0 = chunk_sentences(nlp(document))
    chunks1 = nltk.sent_tokenize(document)
    chunks2 = chunk_markdown_for_rag(document)
    
    print(len(chunks0), len(chunks1), len(chunks2))

    chunks = chunks2
    
    for chunk in chunks:
        documents.append([
            document_id,
            document,
            metadata,
            chunk,
        ])


len(documents)