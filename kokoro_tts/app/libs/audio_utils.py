import numpy as np
from typing import Tuple

def convert_float_to_int16(audio_array: np.ndarray) -> np.ndarray:
    """Convert float audio array to int16 format"""
    # Convert to float32 first to ensure proper scaling
    audio_array = np.array(audio_array, dtype=np.float32)
    # Scale to int16 range (-32768 to 32767)
    return (audio_array * 32767).astype(np.int16)

def get_audio_duration(audio_array: np.ndarray, sample_rate: int = 24000) -> float:
    """Calculate duration of audio in seconds"""
    return len(audio_array) / sample_rate

def format_audio_output(audio_array: np.ndarray, sample_rate: int = 24000) -> Tuple[Tuple[int, np.ndarray], str]:
    """Format audio array for Gradio output with duration info"""
    audio_array = convert_float_to_int16(audio_array)
    duration = get_audio_duration(audio_array, sample_rate)
    return (sample_rate, audio_array), f"Audio Duration: {duration:.2f} seconds"

def concatenate_audio_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    """Concatenate multiple audio chunks into a single array"""
    return np.concatenate(chunks)
