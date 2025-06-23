import os
import importlib.util
import sys
from huggingface_hub import hf_hub_download, snapshot_download
from typing import List, Optional

def load_module_from_file(module_name: str, file_path: str):
    """Load a Python module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def download_model_files(repo_id: str, filenames: List[str], local_dir: Optional[str] = None) -> List[str]:
    """Download multiple files from Hugging Face Hub"""
    paths = []
    for filename in filenames:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            paths.append(path)
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            raise
    return paths

def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't"""
    os.makedirs(path, exist_ok=True)

def find_voice_directory(start_path: str) -> str:
    """Recursively search for directory containing .pt files that don't have 'kokoro' in the name"""
    for root, dirs, files in os.walk(start_path):
        pt_files = [f for f in files if f.endswith('.pt') and 'kokoro' not in f.lower()]
        if pt_files:
            return root
    return ""

def list_voice_files(voices_dir: str) -> List[str]:
    """List available voice files in directory"""
    voices = []
    try:
        # First try the standard locations
        if os.path.exists(os.path.join(voices_dir, 'voices')):
            voice_path = os.path.join(voices_dir, 'voices')
        else:
            voice_path = voices_dir
            
        # If no voices found, try recursive search
        if not os.path.exists(voice_path) or not any(f.endswith('.pt') for f in os.listdir(voice_path)):
            found_dir = find_voice_directory(os.path.dirname(voices_dir))
            if found_dir:
                voice_path = found_dir
                print(f"Found voices in: {voice_path}")
            else:
                print(f"No voice directory found")
                return voices
        
        files = os.listdir(voice_path)
        print(f"Found {len(files)} files in voices directory")
        
        for file in files:
            if file.endswith(".pt") and 'kokoro' not in file.lower():
                voice_name = file[:-3]  # Remove .pt extension
                print(f"Found voice: {voice_name}")
                voices.append(voice_name)
                
        if not voices:
            print("No voice files found in voices directory")
            
    except Exception as e:
        print(f"Error listing voices: {str(e)}")
        import traceback
        traceback.print_exc()
        
    return sorted(voices)

def download_voice_files(repo_id: str, directory: str, local_dir: str) -> None:
    """Download voice files from Hugging Face Hub
    
    Args:
        repo_id: The Hugging Face repository ID
        directory: The directory in the repo to download (e.g. "voices")
        local_dir: Local directory to save files to
    """
    ensure_dir(local_dir)
    try:
        print(f"Downloading voice files from {repo_id}/{directory} to {local_dir}")
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=local_dir,
            allow_patterns=[f"{directory}/*"],
            local_dir_use_symlinks=False
        )
        print(f"Download completed to: {downloaded_path}")
    except Exception as e:
        print(f"Error downloading voice files: {str(e)}")
        import traceback
        traceback.print_exc()
