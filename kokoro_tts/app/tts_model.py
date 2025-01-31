import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple, List
from statistics import mean, median, stdev
from libs import (
    normalize_text,
    chunk_text,
    count_tokens,
    load_module_from_file,
    download_model_files,
    list_voice_files,
    download_voice_files,
    ensure_dir,
    concatenate_audio_chunks
)


class TTSModel:
    """GPU-accelerated TTS model manager"""
    
    def __init__(self):
        self.model = None
        self.voices_dir = "voices"
        self.model_repo = "hexgrad/kLegacy"
        ensure_dir(self.voices_dir)
        self.model_path = None
        
        # Load required modules from v0.19 directory
        py_modules = ["istftnet", "plbert", "models", "kokoro"]
        module_files = download_model_files(self.model_repo, [f"v0.19/{m}.py" for m in py_modules])
        
        for module_name, file_path in zip(py_modules, module_files):
            load_module_from_file(module_name, file_path)
        
        # Import required functions from kokoro module
        kokoro = __import__("kokoro")
        self.generate = kokoro.generate
        self.build_model = __import__("models").build_model
        
    def initialize(self) -> bool:
        """Initialize model and download voices"""
        try:
            print("Initializing model...")
            
            # Download model files
            model_files = download_model_files(
                self.model_repo,
                ["v0.19/kokoro-v0_19.pth", "v0.19/config.json"]
            )
            self.model_path = model_files[0]  # kokoro-v0_19.pth
            
            # Verify local voice files are available
            voices_dir = os.path.join(self.voices_dir, "voices")
            if not os.path.exists(voices_dir):
                raise ValueError("Voice files not found. Please run scripts/download_voices.py first")

            # Verify voices were downloaded successfully
            available_voices = self.list_voices()
            if not available_voices:
                print("Warning: No voices found after initialization")
            else:
                print(f"Found {len(available_voices)} voices")
            
            print("Model initialization complete")
            return True
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return False
    
    def ensure_voice_downloaded(self, voice_name: str) -> bool:
        """Check if specific voice file exists locally"""
        voice_path = os.path.join(self.voices_dir, "voices", f"{voice_name}.pt")
        if not os.path.exists(voice_path):
            print(f"Voice file {voice_name}.pt not found. Please run scripts/download_voices.py first")
            return False
        return True

    def list_voices(self) -> List[str]:
        """List available voices"""
        voices = []
        voices_subdir = os.path.join(self.voices_dir, "voices")
        if os.path.exists(voices_subdir):
            for file in os.listdir(voices_subdir):
                if file.endswith(".pt"):
                    voice_name = file[:-3]
                    voices.append(voice_name)
        return voices
    
    # def _ensure_model_on_gpu(self) -> None:
    #     """Ensure model is on GPU and stays there"""
    #     if not hasattr(self, '_model_on_gpu') or not self._model_on_gpu:
    #         print("Moving model to GPU...")
    #         with torch.cuda.device(0):
    #             torch.cuda.set_device(0)
    #             if hasattr(self.model, 'to'):
    #                 self.model.to('cuda')
    #             else:
    #                 for name in self.model:
    #                     if isinstance(self.model[name], torch.Tensor):
    #                         self.model[name] = self.model[name].cuda()
    #             self._model_on_gpu = True
    
    def _generate_audio(self, text: str, voicepack: torch.Tensor, lang: str, speed: float) -> np.ndarray:
        """GPU-accelerated audio generation"""
        try:
            with torch.cuda.device(0):
                torch.cuda.set_device(0)
                try:
                    # Build model if needed
                    if self.model is None:
                        print("Building model...")
                        device = torch.device('cuda')
                        self.model = self.build_model(self.model_path, device=device)
                        if self.model is None:
                            raise ValueError("Failed to build model")
                        print("Model built successfully")
                    
                    # Move model to GPU if needed
                    if not hasattr(self.model, '_on_gpu'):
                        print("Moving model to GPU...")
                        if hasattr(self.model, 'to'):
                            self.model = self.model.to('cuda')
                        else:
                            for name in self.model:
                                if isinstance(self.model[name], torch.Tensor):
                                    self.model[name] = self.model[name].cuda()
                        self.model._on_gpu = True
                except Exception as e:
                    print(f"Error building model: {str(e)}")
                    print("Attempting to continue")
                    raise e
                # Move voicepack to GPU
                voicepack = voicepack.cuda()
                
                # Run generation with everything on GPU
                audio, _ = self.generate(
                    self.model,
                    text,
                    voicepack,
                    lang=lang,
                    speed=speed
                )
                
                return audio
            
        except Exception as e:
            print(f"Error in audio generation: {str(e)}")
            raise e

    def generate_speech(self, text: str, voice_names: list[str], speed: float = 1.0, gpu_timeout: int = 60, progress_callback=None, progress_state=None, progress=None) -> Tuple[np.ndarray, float]:
        """Generate speech from text. Returns (audio_array, duration)
        
        Args:
            text: Input text to convert to speech
            voice_name: Name of voice to use
            speed: Speech speed multiplier
            progress_callback: Optional callback function(chunk_num, total_chunks, tokens_per_sec, rtf, progress_state, start_time, gpu_timeout, progress)
            progress_state: Dictionary tracking generation progress metrics
            progress: Progress callback from Gradio
        """
        try:
            start_time = time.time()
            with torch.cuda.device(0):
                torch.cuda.set_device(0)
                if not text or not voice_names:
                    raise ValueError("Text and voice name are required")
                            # Build model directly on GPU
                
                # Build model if needed
                if self.model is None:
                    print("Building model...")
                    self.model = self.build_model(self.model_path, device='cuda')
                    if self.model is None:
                        raise ValueError("Failed to build model")
                    print("Model built successfully")
                
                # Move model to GPU if needed
                if not hasattr(self.model, '_on_gpu'):
                    print("Moving model to GPU...")
                    if hasattr(self.model, 'to'):
                        self.model = self.model.to('cuda')
                    else:
                        for name in self.model:
                            if isinstance(self.model[name], torch.Tensor):
                                self.model[name] = self.model[name].cuda()
                    self.model._on_gpu = True
                
                t_voices = []
                if isinstance(voice_names, list) and len(voice_names) > 1:
                    for voice in voice_names:
                        try:
                            voice_path = os.path.join(self.voices_dir, "voices", f"{voice}.pt")
                            try:
                                voicepack = torch.load(voice_path, weights_only=True)
                            except Exception as e:
                                print(f"Warning: weights_only load failed, attempting full load: {str(e)}")
                                voicepack = torch.load(voice_path, weights_only=False)
                            t_voices.append(voicepack)
                        except Exception as e:
                            print(f"Warning: Failed to load voice {voice}: {str(e)}")
                
                    # Combine voices by taking mean
                    voicepack = torch.mean(torch.stack(t_voices), dim=0)
                    voice_name = "_".join(voice_names)
                else:
                    voice_name = voice_names[0]
                    voice_path = os.path.join(self.voices_dir, "voices", f"{voice_name}.pt")
                    try:
                        voicepack = torch.load(voice_path, weights_only=True)
                    except Exception as e:
                        print(f"Warning: weights_only load failed, attempting full load: {str(e)}")
                        voicepack = torch.load(voice_path, weights_only=False)

                # Count tokens and normalize text
                total_tokens = count_tokens(text)
                text = normalize_text(text)
                if not text:
                    raise ValueError("Text is empty after normalization")
                
                # Break text into chunks for better memory management
                chunks = chunk_text(text)
                print(f"Processing {len(chunks)} chunks...")
                
                # Process all chunks within same GPU context
                audio_chunks = []
                chunk_times = []
                chunk_sizes = []  # Store chunk lengths
                total_processed_tokens = 0
                total_processed_time = 0
                
                for i, chunk in enumerate(chunks):
                    chunk_start = time.time()
                    chunk_audio = self._generate_audio(
                        text=chunk,
                        voicepack=voicepack,
                        lang=voice_name[0],
                        speed=speed
                    )
                    chunk_time = time.time() - chunk_start
                    
                    # Calculate per-chunk metrics
                    chunk_tokens = count_tokens(chunk)
                    chunk_tokens_per_sec = chunk_tokens / chunk_time
                    
                    # Update totals for overall stats
                    total_processed_tokens += chunk_tokens
                    total_processed_time += chunk_time
                    
                    # Calculate processing speed metrics
                    chunk_duration = len(chunk_audio) / 24000  # audio duration in seconds
                    rtf = chunk_time / chunk_duration
                    times_faster = 1 / rtf
                    
                    chunk_times.append(chunk_time)
                    chunk_sizes.append(len(chunk))
                    print(f"Chunk {i+1}/{len(chunks)} processed in {chunk_time:.2f}s")
                    print(f"Current tokens/sec: {chunk_tokens_per_sec:.2f}")
                    print(f"Real-time factor: {rtf:.2f}x")
                    print(f"{times_faster:.1f}x faster than real-time")
                    
                    audio_chunks.append(chunk_audio)
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(
                            i + 1,  # chunk_num
                            len(chunks),  # total_chunks
                            chunk_tokens_per_sec,  # Pass per-chunk rate instead of cumulative
                            rtf,
                            progress_state,  # Added
                            start_time,  # Added
                            gpu_timeout,  # Use the timeout value from UI
                            progress  # Added
                        )
            
            # Concatenate audio chunks
            audio = concatenate_audio_chunks(audio_chunks)
    
            # Return audio and metrics
            return (
                audio,  # Audio array
                len(audio) / 24000,  # Duration
                {
                    "chunk_times": chunk_times,
                    "chunk_sizes": chunk_sizes,
                    "tokens_per_sec": [float(x) for x in progress_state["tokens_per_sec"]],
                    "rtf": [float(x) for x in progress_state["rtf"]],
                    "total_tokens": total_tokens,
                    "total_time": time.time() - start_time
                }
            )
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            raise
