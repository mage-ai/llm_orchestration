import json
import httpcore
import time
from typing import List, Union

import numpy as np
import replicate
from replicate.deployment import DeploymentsPredictions

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(documents: List[List[Union[str, List[str]]]], *args, **kwargs):
    use_api = int(kwargs.get('use_api', 0)) == 1
    dimensions = int(kwargs.get('dimensions', 768))

    print(f'use_api: {use_api}')
    print(f'documents: {len(documents)}')

    predictions = []
    for document_id, _document, _metadata, _chunk, tokens in documents:
        print(f'document_id: {document_id}')
        print(f'tokens: {len(tokens)}')

        payload = dict(text_batch=json.dumps(tokens))

        if use_api:
            output = replicate.run(
                'replicate/all-mpnet-base-v2:b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305',
                input=payload,
            )
            predictions.append(output)
        else:
            dp = DeploymentsPredictions(replicate.default_client)
            result = dp.create(
                ('tommydangerous', 'mage-dev2'),
                input=payload,
            )
            predictions.append(result)
            time.sleep(0.5)
            
        print(f'Predictions: {round(100 * len(predictions) / len(documents))}% ({len(predictions)}/{len(documents)})')

    if not use_api:
        def __completed(predictions=predictions) -> int:
            counter = 0
            for prediction in predictions:
                if not prediction.completed_at:
                    prediction.reload()
                
                if prediction.completed_at:
                    counter += 1

            return counter

        def __progress(predictions=predictions) -> float:
            return __completed() / len(predictions)

        progress = __progress()
        while progress < 1.0:
            print(f'Progress: {round(100 * progress)}%')
            progress = __progress()
        
        print(f'Progress: {round(100 * progress)}%')

    documents_more = []
    for tup, prediction in zip(documents, predictions):
        document_id, document, metadata, chunk, tokens = tup

        print(f'document_id: {document_id}')

        if use_api:
            output = prediction
        else:
            output = prediction.output

        if output:
            print(f'output: {len(output)}')
            matrix = [r['embedding'] for r in output]

            # Assuming the 3x768 matrix is stored in a variable called 'matrix'
            vector = np.max(matrix, axis=0)
            print(f'shape: {vector.shape}')
            vector = vector.tolist()
        else:
            print(f'Output is empty for document {document_id}, using another embedding method...')
            
            result = Embedding.create(
                input=json.dumps(reduce_strings_to_length(tokens, 8192)), 
                model='text-embedding-3-large',
                dimensions=dimensions,
                encoding_format='float',
            )
            vector = result.data[0].embedding

        documents_more.append([
            document_id,
            document,
            metadata,
            chunk,
            tokens,
            vector,
        ])

        print(f'{round(100 * len(documents_more) / len(documents))}% ({len(documents_more)}/{len(documents)})')
    
    return [
        documents_more,
    ]