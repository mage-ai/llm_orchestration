import json
import os

from mage_ai.settings.repo import get_repo_path


def load_jsons_and_add_to_list(root_directory):
    combined_list = []
    
    # Walk through the directory, including subdirectories
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".json"):  # Make sure to process only JSON files
                file_path = os.path.join(subdir, file)
                
                # Open and load the JSON file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Assuming the 'data' is a list of items
                for item in data:
                    # Add the file name (without extension) as 'channel' key
                    item['channel'] = os.path.splitext(file)[0]
                    if 'text' in item:
                        combined_list.append(item)
    
    return combined_list


@data_loader
def load_data(*args, **kwargs):
    file_path = os.path.join(get_repo_path(), 'documents', 'slack')
    return [
        load_jsons_and_add_to_list(file_path),
    ]
