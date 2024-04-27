import os
import re
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup

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


def clean_html_and_add_spaces(html_content: str) -> str:
    """
    # Example HTML content
    # Visit Example . Follow us on Twitter !
    html_content = '''
    <p>
        Visit <a href='https://example.com'>Example</a>.
        Follow us on <a href='https://twitter.com/example'>Twitter</a>!
    </p>
    '''

    # This Is a Title This is some text in a paragraph.
    html_content = '''
    <html>

    <head>
        <title>This Is a Title</title>
    </head>

    <body>
        <p>This is some text in a paragraph.</p>
    </body>

    </html>
    '''
    """

    # Use BeautifulSoup to parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all the tags in the document
    for tag in soup.find_all(True):
        # Insert a space before and after each tag in its parent content
        # This helps ensure words don't run together when tags are removed
        tag.insert_before(" ")
        tag.insert_after(" ")

    # Now that spaces have been inserted, get the text without any HTML tags
    clean_text = soup.get_text()

    # Optionally, you might want to clean up multiple consecutive spaces
    # caused by the insertion of spaces around tags
    clean_text = ' '.join(clean_text.split())

    return clean_text
