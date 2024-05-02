import requests
import json
from typing import List, Dict, Any


def construct_combined_prompt(data: List[Dict[str, Any]]) -> str:
    """
    Constructs a single, comprehensive prompt for the LLM based on multiple summaries and topics.
    """
    combined_prompt_parts = []
    for item in data:
        summary, topics = item.get('summary', ''), item.get('topics', [])
        topics_str = ", ".join(topics)
        prompt_part = f"Summary: {summary} Topics: {topics_str}."
        combined_prompt_parts.append(prompt_part)

    combined_prompt = " ".join(combined_prompt_parts) + \
                      " Please adjust the summaries for accuracy and suggest adjusted topics if necessary. " \
                      "Return as a JSON list of dictionaries with keys 'topics', 'summary', " \
                      "'summary_adjusted', and 'topics_adjusted'."
    return combined_prompt


def call_llm_api_with_combined_prompt(prompt: str) -> List[Dict[str, Any]]:
    """
    Calls the LLM API with the combined prompt and returns the API's response.
    """
    url = "https://sorcery.mage.ai:8000/api/v1/generations"
    payload = {
        "model": "deepseek-ai/deepseek-coder-6.7b-base",
        "texts": [prompt]
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        response_data = response.json()
        generation_output = response_data.get('generations', ["{}"])[0]
        try:
            output_data = json.loads(generation_output)
        except json.JSONDecodeError:
            output_data = [{"error": "Failed to decode LLM output as JSON.", "response": generation_output}]
        return output_data
    else:
        return [{"error": "Failed to receive a successful response from the LLM API.", "status_code": response.status_code}]


def update_with_new_keys(existing_data: List[Dict[str, Any]], new_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Updates the existing list of dictionaries with new keys and values from the LLM response.
    """
    # Assuming the order and count of dictionaries in both existing_data and new_data are aligned
    # and correspond to each other.
    for original, update in zip(existing_data, new_data):
        # This loop assumes each dictionary in new_data corresponds to and should update the dictionary in existing_data
        original.update(update)
    return existing_data


@transform
def process(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transforms the input data by calling the LLM API with a single, combined prompt constructed from all items, processing the response,
    and integrating the new data with the existing data.
    """
    prompt = construct_combined_prompt(data)

    new_data = call_llm_api_with_combined_prompt(prompt)

    # Assuming new_data correctly maps back to the original list of dictionaries
    # and contains additional keys for updates.
    # Now, call the update_with_new_keys function to merge the new data with the existing data.
    updated_data = update_with_new_keys(data, new_data)

    return updated_data
