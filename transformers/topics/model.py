import json
from typing import Any, Dict, List, Union

import pandas as pd

from default_repo.llm_orchestration.utils.topic_summary_processor import summarize_and_infer_topics_openai


@transformer
def transform(document: Dict[str, Any], *args, **kwargs) -> pd.DataFrame:
    """
    Transform a single document, represented as a dictionary, to include topics
    and a summary generated from its content.

    :param document: A dictionary containing at least 'document_id', 'document', and 'metadata'.
    :param args: Additional positional arguments (unused in this example).
    :param kwargs: Additional keyword arguments (unused in this example).
    :return: Updated document dictionary including 'topics' and 'summary'.
    """
    document_id = document.get('document_id', '')
    doc_text = document.get('document', '')
    metadata = document.get('metadata', {})

    print(document_id)

    rows = []
    topics = {}
    
    responses = summarize_and_infer_topics_openai(doc_text, verbosity=2)
    for res in responses:
        chunks = list(json.loads(res['choices'][0]['message']['content']).values())[0]
        print(f'chunks: {len(chunks)}')

        for topic_chunk in chunks:
            print(topic_chunk)
            chunk = topic_chunk.get('sentence') or topic_chunk.get('text')
            topic = topic_chunk['topic']
            
            topics[topic] = topics.get(topic) or []
            topics[topic].append(chunk)

            row = dict(chunk=chunk, topic=topic)
            row.update(document)
            rows.append(row)

    print(f'rows:   {len(rows)}')
    print(f'topics: {len(topics)}')
    for topic, chunks in topics.items():
        print(f'\t{topic}: {len(chunks)}')

    df = pd.DataFrame(rows)

    return df
