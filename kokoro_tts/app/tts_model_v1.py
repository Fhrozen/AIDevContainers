import os
import torch
import numpy as np
import time
from typing import Tuple, List
from kokoro import KPipeline


class TTSModelV1:
    """KPipeline-based TTS model for v1.0.0"""
    
    def __init__(self):
        self.pipeline = None
        self.voices_dir = os.path.join(os.path.dirname(__file__), "voices_v1")
        
    def initialize(self) -> bool:
        """Initialize KPipeline"""
        try:
            print("Initializing v1.0.0 model...")
            self.pipeline = None # cannot be initialized outside of GPU decorator
            print("Model initialization complete")
            return True
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return False
    
    def list_voices(self) -> List[str]:
        """List available voices from voices_v1 directory"""
        voices = []
        if os.path.exists(self.voices_dir):
            for file in os.listdir(self.voices_dir):
                if file.endswith(".pt"):
                    voice_name = file[:-3]
                    voices.append(voice_name)
        return sorted(voices)

    def generate_speech(self, text: str, voice_names: list[str], speed: float = 1.0, gpu_timeout: int = 60, progress_callback=None, progress_state=None, progress=None) -> Tuple[np.ndarray, float]:
        """Generate speech from text using KPipeline
        
        Args:
            text: Input text to convert to speech
            voice_names: List of voice names to use (will be mixed if multiple)
            speed: Speech speed multiplier
            progress_callback: Optional callback function
            progress_state: Dictionary tracking generation progress metrics
            progress: Progress callback from Gradio
        """
        try:
            start_time = time.time()
            if self.pipeline is None:
                lang_code = voice_names[0][0] if voice_names else 'a'
                self.pipeline = KPipeline(lang_code=lang_code)
                
            if not text or not voice_names:
                raise ValueError("Text and voice name are required")
            
            # Handle voice selection
            if isinstance(voice_names, list) and len(voice_names) > 1:
                # For multiple voices, join them with underscore
                voice_name = "_".join(voice_names)
            else:
                voice_name = voice_names[0]
            
            # Initialize tracking
            audio_chunks = []
            chunk_times = []
            chunk_sizes = []
            total_tokens = 0
            
            # Preprocess text - replace single newlines with spaces while preserving paragraphs
            processed_text = '\n\n'.join(
                paragraph.replace('\n', ' ').replace('  ', ' ').strip()
                for paragraph in text.split('\n\n')
            )
            
            # Get generator from pipeline
            generator = self.pipeline(
                processed_text,
                voice=voice_name,
                speed=speed,
                split_pattern=r'\n\n+'  # Split on double newlines or more
            )
            
            # Process chunks
            total_duration = 0  # Total audio duration in seconds
            total_process_time = 0  # Total processing time in seconds
            
            for i, (gs, ps, audio) in enumerate(generator):
                chunk_process_time = time.time() - start_time - total_process_time
                total_process_time += chunk_process_time
                audio_chunks.append(audio)
                
                # Calculate metrics
                chunk_tokens = len(gs)
                total_tokens += chunk_tokens
                
                # Calculate audio duration
                chunk_duration = len(audio) / 24000  # Convert samples to seconds
                total_duration += chunk_duration
                
                # Calculate speed metrics
                tokens_per_sec = chunk_tokens / chunk_duration  # Tokens per second of audio
                rtf = chunk_process_time / chunk_duration  # Real-time factor
                
                chunk_times.append(chunk_process_time)
                chunk_sizes.append(chunk_tokens)
                
                print(f"Chunk {i+1}:")
                print(f"  Process time: {chunk_process_time:.2f}s")
                print(f"  Audio duration: {chunk_duration:.2f}s")
                print(f"  Tokens/sec: {tokens_per_sec:.1f}")
                print(f"  Real-time factor: {rtf:.3f}")
                print(f"  Speed: {(1/rtf):.1f}x real-time")
                
                # Update progress
                if progress_callback and progress_state:
                    # Initialize lists if needed
                    if "tokens_per_sec" not in progress_state:
                        progress_state["tokens_per_sec"] = []
                    if "rtf" not in progress_state:
                        progress_state["rtf"] = []
                    if "chunk_times" not in progress_state:
                        progress_state["chunk_times"] = []
                    
                    # Update progress state
                    progress_state["tokens_per_sec"].append(tokens_per_sec)
                    progress_state["rtf"].append(rtf)
                    progress_state["chunk_times"].append(chunk_process_time)
                    
                    progress_callback(
                        i + 1,
                        -1,  # Let UI handle total chunks
                        tokens_per_sec,
                        rtf,
                        progress_state,
                        start_time,
                        gpu_timeout,
                        progress
                    )
            
            # Concatenate audio chunks
            audio = np.concatenate(audio_chunks)

            # Return audio and metrics
            return (
                audio,
                len(audio) / 24000,
                {
                    "chunk_times": chunk_times,
                    "chunk_sizes": chunk_sizes,
                    "tokens_per_sec": [float(x) for x in progress_state["tokens_per_sec"]] if progress_state else [],
                    "rtf": [float(x) for x in progress_state["rtf"]] if progress_state else [],
                    "total_tokens": total_tokens,
                    "total_time": time.time() - start_time
                }
            )
            
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            raise
