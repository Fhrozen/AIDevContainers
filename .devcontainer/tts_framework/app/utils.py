import os
from huggingface_hub import (
    snapshot_download
)


def download_files(repo_id: str, directory: str, local_dir: str, pattern: str = "*") -> None:
    """Download voice files from Hugging Face Hub
    
    Args:
        repo_id: The Hugging Face repository ID
        directory: The directory in the repo to download (e.g. "voices")
        local_dir: Local directory to save files to
    """
    os.makedirs(local_dir, exist_ok=True)
    if directory == ".":
        _patern = pattern
    else:
        _patern = f"{directory}/{pattern}"
    try:
        print(f"Downloading files from {repo_id}/{directory} to {local_dir}")
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=local_dir,
            allow_patterns=[_patern],
            local_dir_use_symlinks=False
        )
        print(f"Download completed to: {downloaded_path}")
    except Exception as e:
        print(f"Error downloading files: {str(e)}")
        import traceback
        traceback.print_exc()
