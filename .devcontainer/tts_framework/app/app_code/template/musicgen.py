import os
from typing import Generator, List, Optional, Tuple
import time
import re
import glob

import logging

import librosa
import numpy as np
import torch

import onnxruntime
from onnxruntime import InferenceSession

from transformers import AutoTokenizer
from utils import download_files

import soundfile as sf

from transformers import AutoProcessor, MusicgenForConditionalGeneration


class MusicGenTransformers:
    """MusicGen model for onnx"""

    def __init__(self):
        dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.models_dir = os.path.join(os.path.dirname(__file__), "downloaded_mgmodels", "onnx")
        self.configs_dir = os.path.join(os.path.dirname(__file__), "downloaded_mgconfigs")
        self.repo_id = "xenova/musicgen-small"
        self.model = None
        self.tokenizer = None
        self.samplerate = 32000
        self.modules_names = ["build_delay", "decoder_model_merged", "encodec", "text_encoder"]
        self.frame_rate = 50

    def initialize(self) -> bool:
        """Initialize"""
        try:
            print("Initializing model...")
            # self.set_flavor()
            # self.set_tokenizer()
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
            cache_dir = self.configs_dir,
        )
        model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small",
            torch_dtype=torch.bfloat16, 
            device_map=device
        )
        model.generation_config.cache_implementation = "static"
        # model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        self.model = model

    def set_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
        self.unkown_token = -1
        self.pad_token_id = 2048
        self.decoder_start_token_id = 2048
        self.bos_token_id = 2048
        self.num_codebooks = 4
        
        self.num_decoder_layers = 24
        return

    def _list_flavors(self) -> List[str]:
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

    def _set_flavor(self, flavor: Optional[str] = None):
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
        output_names = [output.name for output in modules["decoder_model_merged"].get_outputs()]
        self.dec_output_names = [x for x in output_names if x != "logits"]
        # self.decoder = ORTDecoderForSeq2Seq(modules["decoder_model_merged"], self)

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

    def _generate(
        self,
        text: str,
        audio: np.ndarray | None = None,
        duration: int = 10,
        guidance: int | None = 1,
        temperature: float = 1.0,
        use_sampling: bool = True,
        top_k: int = 250,
        top_p: float = 0.0,
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
        encoder_outputs = self.modules["text_encoder"].run(None, {**inputs_ids})[0]
        encoder_attn_mask = np.ones([1, encoder_outputs.shape[1]], dtype=np.long)

        # Generate codes
        if guidance is not None and guidance > 1:
            encoder_outputs = np.concat([encoder_outputs, np.zeros_like(encoder_outputs)], axis=0)
        
        decoder_input_ids = np.full(
            [1 * self.num_codebooks, 16],
            -1,
            dtype=np.long
        )
        decoder_input_ids[:, 0] = self.bos_token_id
        
        # Build and apply delay mask
        max_len = duration * self.frame_rate + 1
        decoder_input_ids, delay_mask = self.modules["build_delay"].run(None, {
            "input_ids": decoder_input_ids,
            "pad_token_id": np.full([1], self.pad_token_id),
            "max_length": np.full([1], max_len),
        })
        decoder_input_ids = np.pad(
            decoder_input_ids,
            [[0, 0], [0, max_len - 1]],
            mode="constant",
            constant_values=-1
        )

        # gen_codes = 
        decoder_cache = {}
        for key in self.dec_output_names:
            _key = key.replace("present", "past_key_values")
            decoder_cache[_key] = None

        output_names = ["logits"] + self.dec_output_names
        use_cache = np.zeros([1], dtype=bool)
        for idx in range(1, max_len):
            _input_ids_frame = decoder_input_ids[:, :idx]
            # print(_input_ids_frame.shape)
            outputs = self.modules["decoder_model_merged"].run(output_names, {
                "encoder_attention_mask": encoder_attn_mask,
                "encoder_hidden_states": encoder_outputs,
                "use_cache_branch": use_cache,
                "input_ids": _input_ids_frame,
                **decoder_cache
            })
            _logits = outputs[0]
            next_tokens = np.argmax(_logits[:, -1], axis=-1, keepdims=True)
            decoder_input_ids[:, idx:idx + 1] = next_tokens
            for kdx, key in enumerate(self.dec_output_names):
                _key = key.replace("present", "past_key_values")
                _value = decoder_cache[_key]  # outputs[kdx + 1]  # 
                # if idx > 1 and "encoder" in _key:
                #     continue
                if _value is None:
                    _value = outputs[kdx + 1]
                else:
                    if outputs[kdx + 1].shape[0] == 0:
                        continue
                    if "encoder" in key:
                        continue
                    last_qv = outputs[kdx + 1][:, :, -1:]
                    _value = np.concat([_value, last_qv], axis=2)
                decoder_cache[_key] = _value
                if kdx == 0:
                    print(_key, _value.shape)
                # print(_key, _value.shape)
                # if idx > 1 and "encoder" in _key:
                #     continue
                # decoder_cache[_key] = outputs[kdx + 1]
            # print(decoder_cache[_key].shape)
            

            if not np.any(use_cache):
                use_cache = np.ones([1], dtype=bool)

        decoder_input_ids = np.where(delay_mask == -1, decoder_input_ids, delay_mask)[:, 1:]
        
        generated = self.modules["encodec"].run(None, {
            "audio_codes": decoder_input_ids[None, None]
        })[0]
        
        sf.write("out.wav", generated[0, 0], self.samplerate)
        print(generated.shape)
        exit()

        # print(logits)
        
        return

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
        # print(inputs)
        # exit()
        max_new_tokens = duration * self.frame_rate

        audio_values = self.model.generate(
            **inputs,
            do_sample=use_sampling,
            guidance_scale=guidance,
            max_new_tokens=max_new_tokens
        ).squeeze().to("cpu").float().numpy()
        return audio_values, len(audio_values) / self.samplerate, None

if __name__ == "__main__":
    tts_model = MusicGenTransformers()
    tts_model.initialize()
    # print(tts_model.list_flavors())
    command = "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130"
    wav = tts_model.generate(command)
    # print(wav)
