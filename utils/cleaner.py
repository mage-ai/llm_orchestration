import os
import re
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from mage_ai.shared.strings import replacer


def clean_name(
    name,
    allow_characters: List[str] = None,
    allow_number: bool = False,
    case_sensitive: bool = False,
):
    """
    Clean a name by removing unwanted characters and replacing them with underscores,
    while optionally preserving specified allowed characters and handling numbers at
    the beginning of the name.

    Args:
        name (str): The name to be cleaned.
        allow_characters (List[str], optional): List of allowed characters to preserve.
            Defaults to None.
        allow_number (bool, optional): Whether to allow numbers at the beginning of
            the name. Defaults to False.
        case_sensitive (bool, optional): Whether to preserve the case of the name.
            Defaults to False.

    Returns:
        str: The cleaned name.
    """
    if allow_characters is None:
        allow_characters = []
    # Remove unwanted characters
    for c in ['\ufeff', '\uFEFF', '"', '$', '\n', '\r', '\t']:
        name = name.replace(c, '')

    indexes_of_allowed_characters = {}
    for allowed_char in allow_characters:
        if allowed_char not in indexes_of_allowed_characters:
            indexes_of_allowed_characters[allowed_char] = []

        for idx, char in enumerate(name):
            if char == allowed_char:
                indexes_of_allowed_characters[allowed_char].append(idx)

    # Replace space with underscore
    name = re.sub(r'\W', '_', name)

    for allowed_char, indexes in indexes_of_allowed_characters.items():
        for idx in indexes:
            name = replacer(name, allowed_char, idx)

    if name and not allow_number and re.match(r'\d', name[0]):
        name = f'letter_{name}'

    # If the column name is not case sensitive, use the lower case of it
    return name.lower() if not case_sensitive else name