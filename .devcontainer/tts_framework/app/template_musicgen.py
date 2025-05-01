import os
from typing import Generator, List, Optional, Tuple
import time
import re
import glob

import logging

import librosa
import numpy as np

import onnxruntime
from onnxruntime import InferenceSession

from transformers import AutoTokenizer
from utils import download_files


class MusicGenONNX:
    """MusicGen model for onnx"""

    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(__file__), "downloaded_mgmodels", "onnx")
        self.configs_dir = os.path.join(os.path.dirname(__file__), "downloaded_mgconfigs")
        self.repo_id = "xenova/musicgen-small"
        self.model = None
        self.tokenizer = None
        self.samplerate = 32000
        self.modules_names = ["build_delay", "decoder_model_merged", "encodec", "text_encoder"]

    def initialize(self) -> bool:
        """Initialize"""
        try:
            print("Initializing model...")
            self.set_flavor()
            self.set_tokenizer()
            print("Model initialization complete")
            return True
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return False

    def set_tokenizer(self):
        configs_dir = self.configs_dir
        check_fn = os.path.join(configs_dir, ".done")
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
        return

    
    def list_flavors(self) -> List[str]:
        """List available models from voices_v1 directory"""
        _models_dir = os.path.dirname(self.models_dir)
        check_fn = os.path.join(_models_dir, ".done")
        if not os.path.exists(check_fn):
            for module in self.modules_names:
                download_files(self.repo_id, "onnx", _models_dir, f"{module}*")
            with open(check_fn, "w") as writer:
                writer.write("\n")

        flavors = []
        if os.path.exists(self.models_dir):
            listfiles = glob.glob(os.path.join(self.models_dir, "text_encoder*.onnx"))
            for filename in listfiles:
                model_name = filename.split(".onnx")[0]
                model_name = model_name.split("text_encoder")[1].lstrip("_")
                model_name = "model" if len(model_name) == 0 else f"model_{model_name}"
                flavors.append(os.path.basename(model_name))
        return flavors

    def set_flavor(self, flavor: Optional[str] = None):
        if flavor is None:
            flavor = self.list_flavors()[0]

        use_gpu = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        providers = ["CUDAExecutionProvider"] if use_gpu else []
        providers.append("CPUExecutionProvider")
        flavor = flavor.replace("model", "")

        sess_options = onnxruntime.SessionOptions()
        # sess_options.log_severity_level=1
        modules = {}
        for mod_name in self.modules_names:
            model_path = os.path.join(self.models_dir, f"{mod_name}*{flavor}.onnx")
            model_path = glob.glob(model_path)[0]
            modules[mod_name] = InferenceSession(model_path, sess_options, providers=providers)
        self.modules = modules

    def extra_tokenize(self, tokens):
        tks = []
        tokens = "+".join(tokens)
        pcount = 0

        for t in tokens.split(" "):
            t = t.lstrip("+").split("+")[:-1] + [' ']
            next_ps = t
            next_pcount = pcount + len(next_ps)
            if next_pcount > 510:
                z = len(tks) - len(next_ps)
                ps = "".join(tks[:z]).strip()
                yield ps, ps, None
                tks = tks[z:]
                pcount = len(tks)
                if not tks:
                    next_ps = next_ps.lstrip()
            tks.extend(t)
            pcount += len(next_ps)
        if tks:
            ps = "".join(tks)
            yield ps.strip(), ps.strip(), None

    def infer(self, phonemes, speed = 1.0):
        voice = self.voice
        tokens = [i for i in map(self.VOCAB.get, phonemes) if i is not None]
        _lenght = len(tokens)
        _lenght = _lenght if _lenght < 510 else 509
        voice = self.voice[_lenght]
        tokens = [[0, *tokens, 0]]
        audio = self.model.run(None, {
            "input_ids" : tokens,
            "style": voice,
            "speed": np.ones(1, dtype=np.float32) * speed
        })[0]
        if audio.shape[0] == 1:
            audio = audio[0]
        return audio

    def generate(
        self,
        text: str,
        duration: int = 1,
        guidance: int = 1,
        temperature: float = 1.0,
        gpu_timeout: int = 60,
        progress_callback=None,
        progress_state=None,
        progress=None,
        split_pattern=r'\n\n+',
        trim=True,
    ) -> Tuple[np.ndarray, float]:
        """Generate speech from text using KPipeline
        
        Args:
            text: Input text to convert to speech
            voice_names: List of voice names to use (will be mixed if multiple)
            speed: Speech speed multiplier
            progress_callback: Optional callback function
            progress_state: Dictionary tracking generation progress metrics
            progress: Progress callback from Gradio
        """
        if not text:
            raise ValueError("Text and voice name are required")

        # Pre
        inputs_ids = self.tokenizer(text, return_tensors="np")
        tokens = self.modules["text_encoder"].run(None, {**inputs_ids})[0]
        print(tokens)
        return


if __name__ == "__main__":
    tts_model = MusicGenONNX()
    tts_model.initialize()
    print(tts_model.list_flavors())
    command = "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130"
    wav = tts_model.generate(command)
    print(wav)
