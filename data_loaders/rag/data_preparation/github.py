import os
import requests

from mage_ai.settings.repo import get_repo_path

def download_repo(repo_url, local_dir):
    """
    Download all files from a GitHub repository.

    Args:
        repo_url (str): The URL of the GitHub repository.
        local_dir (str): The local directory to save the downloaded files.
    """
    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)

    # Download the repository ZIP archive
    repo_zip_url = f"{repo_url}/archive/master.zip"
    response = requests.get(repo_zip_url)
    
    # Save the ZIP archive to a temporary file
    with open("repo.zip", "wb") as f:
        f.write(response.content)
    
    # Extract the ZIP archive to the local directory
    import zipfile
    with zipfile.ZipFile("repo.zip", "r") as zip_ref:
        zip_ref.extractall(local_dir)
    
    # Remove the temporary ZIP archive
    os.remove("repo.zip")

    print(f"Repository downloaded to {local_dir}")


@data_loader
def transform_custom(*args, **kwargs):
    repo_url = "https://github.com/mage-ai/mage-ai"
    local_dir = os.path.join(get_repo_path(), 'documents', 'code')
    download_repo(repo_url, local_dir)

    subfolder = kwargs.get('subfolder')
    if subfolder:
        local_dir = os.path.join(local_dir, subfolder)

    return [
        local_dir,
    ]