import re

def remove_markdown_keep_text(text: str) -> str:
    """
    Removes Markdown special characters, aiming to preserve the plain text.
    """
    patterns = [
        # Remove Markdown links and images, leaving alt text or link text
        (r'\!\[(.*?)\]\(.*?\)', r'\1'),  # Image descriptions
        (r'\[(.*?)\]\(.*?\)', r'\1'),    # Link text
        # Remove emphasis, strong, strikethrough decoration, leaving only the text
        (r'\*\*(.*?)\*\*|__(.*?)__', r'\1\2'),  # Bold
        (r'\*(.*?)\*|_(.*?)_', r'\1\2'),        # Italic
        (r'\~\~(.*?)\~\~', r'\1'),              # Strikethrough
        # Remove inline code, blockquotes, code blocks (keeping contained text, though code formatting will be lost)
        (r'`{3,}[\s\S]*?`{3,}|`(.+?)`', r'\1'), # Inline code and code blocks
        (r'^\s*> ?(.*)', r'\1'),                # Blockquotes
        # Headers, lists (ordered, unordered), horizontal rules
        (r'^\s*#{1,6} ?(.*)', r'\1'),           # Headers
        (r'^\s*[*+-] ?(.*)', r'\1'),            # Unordered lists
        (r'^\s*\d+\. ?(.*)', r'\1'),            # Ordered lists
        (r'^\s*([-*_]\s*){3,}\s*$', ''),        # Horizontal rules
    ]

    clean_text = text
    for pattern, replacement in patterns:
        clean_text = re.sub(pattern, replacement, clean_text, flags=re.MULTILINE)

    # Further clean up to normalize excessive newlines introduced by removing Markdown elements
    clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text).strip()

    return clean_text


def remove_markdown_metadata(text: str) -> str:
    """
    Removes YAML front matter from the given text.
    """
    # Regular expression to match YAML front matter. It looks for the pattern that starts and ends with three dashes (---)
    # and contains any characters in between, non-greedily (?s enables '.' to match newline characters as well)
    pattern = r'^---(?s).*?^---\s*'

    # Replace the YAML front matter with an empty string if found
    cleaned_text = re.sub(pattern, '', text, flags=re.MULTILINE)

    return cleaned_text.strip()
