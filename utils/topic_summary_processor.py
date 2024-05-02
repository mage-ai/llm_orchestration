import json
import os
import re
import requests
from typing import Any, Callable, Dict, List, Optional, Union

import anthropic
import pandas as pd
import spacy

from default_repo.llm_orchestration.utils.markdown import remove_markdown_keep_text, remove_markdown_metadata
from default_repo.llm_orchestration.utils.chunking import chunk_sentences
from default_repo.llm_orchestration.utils.openai import OpenAIChatCompletion


def replace_html_tags_with_spaces(text: str) -> str:
    # Define the regex pattern for matching HTML tags
    pattern = r'<[^>]+>'
    # Replace all occurrences of the pattern with a space
    cleaned_text = re.sub(pattern, ' ', text)
    return cleaned_text


def parse_output(output: str) -> List[Dict[str, str]]:
    # Regular expression to match <topic> and <chunk> contents
    topic_pattern = r'<topic>(.*?)<\/topic>'
    chunk_pattern = r'<chunk>(.*?)<\/chunk>'

    # Find all matches in the output
    topics = re.findall(topic_pattern, output)
    chunks = re.findall(chunk_pattern, output)

    # Combine topics and chunks into a list of dictionaries
    result = [{'topic': topic, 'chunk': chunk} for topic, chunk in zip(topics, chunks)]

    return result


def bucket_sentences_by_max_tokens(sentences: List[str], max_tokens: int = 4096):
    """
    Bucket sentences by a maximum token size approximated by character count.

    :param sentences: List of sentences to be bucketed.
    :param max_tokens: Maximum token size (character count) per bucket.
    :return: A list of lists, each sub-list containing sentences that fit within the max token size.
    """
    buckets = []
    current_bucket = []
    current_count = 0

    for sentence in sentences:
        # Approximate token size by character count for this example
        sentence_size = len(sentence)

        # Check if adding this sentence would exceed the max token size
        if current_count + sentence_size <= max_tokens:
            current_bucket.append(sentence)
            current_count += sentence_size
        else:
            # Start a new bucket if the current one is full
            if current_bucket:  # Avoid adding an empty bucket
                buckets.append(current_bucket)
            current_bucket = [sentence]  # Start new bucket with the current sentence
            current_count = sentence_size

    # Don't forget to add the last bucket if it isn't empty
    if current_bucket:
        buckets.append(current_bucket)

    return buckets


def redistribute_items_evenly(list_of_lists):
    # Flatten the list of lists while preserving order
    flattened_items = [item for sublist in list_of_lists for item in sublist]
    total_items = len(flattened_items)

    # Calculate the number of sub-lists to determine how to redistribute items evenly
    num_lists = len(list_of_lists)

    # Calculate the new size for all sub-lists (minimal size)
    min_items_per_list = total_items // num_lists

    # Calculate the number of sub-lists that should have one more item (to handle remainder items)
    num_lists_with_extra_item = total_items % num_lists

    # Initialize variables to help with redistribution
    new_list_of_lists = []
    current_index = 0

    # Distribute items evenly, respecting original order
    for _ in range(num_lists):
        # Determine the size of the current sub-list
        current_sublist_size = min_items_per_list
        if num_lists_with_extra_item > 0:
            current_sublist_size += 1
            num_lists_with_extra_item -= 1

        # Extract the current sub-list from the flattened list
        current_sublist = flattened_items[current_index:current_index + current_sublist_size]
        new_list_of_lists.append(current_sublist)

        # Update the current index
        current_index += current_sublist_size

    return new_list_of_lists


def list_to_dicts(input_list: List[str]) -> List[Dict[str, str]]:
    result = []
    for i in range(0, len(input_list), 2):
        # Ensure there's a pair (sentence and topic)
        if i+1 < len(input_list):
            sentence, topic = input_list[i], input_list[i+1]
            result.append({'topic': topic, 'chunk': sentence})
    return result


def extract_error_context(e: json.JSONDecodeError, text: str, surround=10):
    # Extract the exact position of the error
    pos = e.pos
    # Calculate start and end positions for the context
    start = max(pos - surround, 0)
    end = min(pos + surround, len(text))

    # Extract the context from the text with the error marked
    context = text[start:end]
    return context


def iterate_document(document: str, verbosity: int = 0):
    nlp = spacy.load('en_core_web_lg')
    document = remove_markdown_metadata(document)
    document = remove_markdown_keep_text(document)
    document = replace_html_tags_with_spaces(document)
    buckets = bucket_sentences_by_max_tokens(chunk_sentences(nlp(document)))
    buckets = redistribute_items_evenly(buckets)

    if verbosity >= 1:
        print(f'Number of buckets: {len(buckets)}')

    arr = []
    for idx, sentences in enumerate(buckets):
        # Constructing the prompt for summarization and topic inference
        sentences_context = """
            <sentences>
            {}
            </sentences>
            """.format('\n'.join([f'<sentence>{sentence}</sentence>' for sentence in sentences]))

        example = """
        <example>
        <topic>Topic 1</topic>
        <chunk>Sentence 1</chunk>
        <topic>Topic 2</topic>
        <chunk>Sentence 2</chunk>
        </example>
        """

        def __build_instructions(text: str = '', example: Optional[str] = None) -> str:
            instructions = [
                text,
                'Read each sentence inside the <sentence> tag.',
                'Infer the topic of each sentence based on all topics from <sentences>.',
            ]

            if example:
                instructions.append('Follow the structure in the <example>.')

            return """
            <instructions>
            {}
            </instructions>
            """.format('\n'.join(instructions))

        if verbosity >= 2:
            print(f'Bucket {idx+1}: {len(sentences)} sentences')

        yield dict(
            build_instructions=__build_instructions,
            document=document,
            example=example,
            sentences=sentences,
            sentences_context=sentences_context,
        )


def summarize_and_infer_topics_openai(
    document: str,
    sample: Optional[int] = None,
    verbosity: int = 0,
) -> List[Dict[str, str]]:
    arr = []
    counter = 0
    for config in iterate_document(document, verbosity=verbosity):
        sentences_context: str = config['sentences_context']
        build_instructions: Callable = config['build_instructions']

        instructions: str = build_instructions("""
        You are a helpful assistant designed to output JSON.
        Infer the topic from the content sent to you.
        The topic is a technical topic from a technical documentation for
        data engineers using Mage AI (www.mage.ai).
        """, example=None)

        messages = [
            {
                "role": "system",
                "content": '\n'.join([
                    sentences_context,
                ]),
            },
            {
                "role": "user",
                "content": instructions,
            }
        ]

        resp = OpenAIChatCompletion.post(messages)
        arr.append(resp)

        counter += 1

        if sample is not None and sample >= counter:
            break

    return arr


def summarize_and_infer_topics_anthropic(document: str, sample: Optional[int] = None, verbosity: int = 0) -> List[Dict[str, str]]:
    # Getting the Anthropics SDK environment and authentication from environment variables
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_api_key:
        return [{'error': 'The ANTHROPIC_API_KEY environment variable is not set.'}]

    anthropic_api = anthropic.Anthropic(api_key=anthropic_api_key)

    arr = []
    counter = 0

    for config in iterate_document(document, verbosity=verbosity):
        document = config['document']
        sentences_context: str = config['sentences_context']
        example: str = config['example']
        build_instructions: Callable = config['build_instructions']
        instructions: str = build_instructions("""
        Wrap the topic in the XML tag <topic> and the sentence in the XML tag <chunk>.
        Return all the tags and their contents in a string with newlines separating each pair.
        """)

        response = anthropic_api.messages.create(
            model='claude-3-opus-20240229',
            system='\n'.join([
                example,
                instructions,
            ]),
            max_tokens=4096,
            messages=[
                dict(role='user', content=sentences_context),
                dict(role='assistant', content='The output starts here:'),
            ],
        ).to_dict()
        response_text = response.get('content', [{}])[0].get('text')
        arr += parse_output(response_text)

        counter += 1

        if sample is not None and counter >= sample:
            break

    return arr


def summarize_and_infer_topics(document):
    # Define the URL of the API
    url = 'https://sorcery.mage.ai:8000/api/v1/generations'

    # Prepare the API payload, instructing the LLM on what to generate
    payload = {
        'model': 'deepseek-ai/deepseek-coder-6.7b-base',
        'texts': [
            f"Given the document: '{document}'. Infer the main topics about the "
            "document and then summarize the text into a few paragraphs or sentences. "
            "Format your answer as a JSON string with the keys 'topics' and 'summary'."
        ],
    }

    # Convert the payload to JSON format
    headers = {'Content-Type': 'application/json'}

    # Send the POST request to the LLM API
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Assuming the LLM responds with the required output structure
    if response.status_code == 200:
        # Parse the JSON response
        response_data = response.json()

        # Extract the generation output
        generation_output = response_data.get('generations', ["{}"])[0]

        # Attempt to parse the generation output as JSON
        try:
            output_data = json.loads(generation_output)
        except json.JSONDecodeError:
            # If the output is not a valid JSON string, return an error message
            output_data = {
                'error': 'Failed to decode LLM output as JSON.',
                'response': generation_output
            }

        return output_data
    else:
        # If the response status code is not 200 (OK), return an error message
        return {
            'error': 'Failed to receive a successful response from the LLM API.',
            'status_code': response.status_code
        }
