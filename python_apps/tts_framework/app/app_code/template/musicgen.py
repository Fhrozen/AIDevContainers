import os
# from typing import Generator, List, Optional, Tuple
# import time
# import re
# import glob

# import logging

# import librosa
import numpy as np
import torch

# import onnxruntime
# from onnxruntime import InferenceSession
# import soundfile as sf
from transformers import (
    # AutoTokenizer,
    AutoProcessor,
    MusicgenForConditionalGeneration
)

from .base import SynthesizerBase
from .utils.common import download_files


class MusicGenTransformers(SynthesizerBase):
    """MusicGen model for onnx"""

    def __init__(self):
        self.models_dir = os.path.join(self.dirpath, "onnx")
        self.repo_id = "xenova/musicgen-small"
        self.samplerate = 32000
        self.modules_names = ["build_delay", "decoder_model_merged", "encodec", "text_encoder"]
        self.frame_rate = 50

    def initialize(self) -> bool:
        """Initialize"""
        try:
            print("Initializing model...")
            self.set_model()
            print("Model initialization complete")
            return True
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return False

    def set_model(self):
        try:
            device = torch.cuda.device_count() - 1
            device = f"cuda:{device}"
        except:
            device = "cpu"
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            "facebook/musicgen-small",
            cache_dir = self.dirpath,
        )
        model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small",
            torch_dtype=torch.bfloat16, 
            device_map=device
        )
        model.generation_config.cache_implementation = "static"
        # model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        self.model = model

    def gen_args(self):
        self._keys = ["duration", "guidance", "use_sampling"]
        self._types = ["slider", "slider", "toogle"]
        self._labels = [
            "Duration",
            "Guidance scale",
            "Use Sampling"
        ]
        self._values = [10, 3, True]
        self._kwargs = [
            {"min_value": 1, "max_value": 30, "step": 1},
            {"min_value": 1, "max_value": 30, "step": 1},
            {}
        ]

    def generate(
        self,
        text: str,
        audio: np.ndarray | None = None,
        duration: int = 30,
        guidance: int | None = 3,
        temperature: float = 1.0,
        use_sampling: bool = True,
        top_k: int = 250,
        top_p: float = 0.0,
    ):
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.device)
        print(inputs['input_ids'])
        exit()
        max_new_tokens = duration * self.frame_rate

        audio_values = self.model.generate(
            **inputs,
            do_sample=use_sampling,
            guidance_scale=guidance,
            max_new_tokens=max_new_tokens
        ).squeeze().to("cpu").float().numpy()
        return audio_values, len(audio_values) / self.samplerate, None


def _main():
    tts_model = MusicGenTransformers()
    tts_model.initialize()
    # print(tts_model.list_flavors())
    command = "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130"
    wav = tts_model.generate(command)
    print(wav)


if __name__ == "__main__":
    _main()

