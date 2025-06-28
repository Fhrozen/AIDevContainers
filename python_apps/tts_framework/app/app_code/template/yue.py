import glob
import json
import os
from typing import List, Optional, Tuple
import re

from numba import cuda
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer, PretrainedConfig
import soundfile as sf

from optimum.onnxruntime import ORTModelForCausalLM
from .base import SynthesizerBase
from .utils.common import download_files
from .utils.yue_mmtkn import MMSentencePieceTokenizer
from .utils.yue_codec import CodecManipulator


LANG_CODES = {
    "en": "American English",
    "jp": "Japanese",
    "kr": "Korean"
}


def split_lyrics(lyrics: str):
    pattern = r"\[(\w+)\]\s*(.*?)(?=\s*\n\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
    return structured_lyrics


class YueTTSONNX(SynthesizerBase):
    def __init__(self):
        self.working_dir = os.path.join(self.dirpath, "yue")
        self.repo_id = {
            "s1_en": "Fhrozen/YuE-s1-7B-anneal-en-cot-ONNX",
            "s1_jp": "Fhrozen/YuE-s1-7B-anneal-jp-kr-cot-ONNX",
            "s2": "Fhrozen/YuE-s2-1B-general-ONNX",
            "codec": "Fhrozen/xcodec-mini-ONNX"
        }
        self.lang_code = None
        self.samplerate = 24000
        try:
            counts = len(cuda.gpus)
        except:
            counts = 0
        self.num_gpus = counts

    def _initialize(self) -> bool:
        """Class Initialize"""
        self.set_language()

    def list_flavors(self) -> List[str]:
        """List available models """
        flavors = ["bnb4", "int8", "q4", "q4f16", "quantized", "uint8"]
        return flavors

    def list_languages(self) -> List[str]:
        """List available voices from voices_v1 directory"""
        langs = list(LANG_CODES.values())
        return langs

    def set_language(self, langid: Optional[str] = None):
        if langid is None:
            code = "en"
        else:
            _codes = {v:k for k, v in LANG_CODES.items()}
            code = _codes[langid]
        if code == self.lang_code:
            return
        self.unload_model()
        self.lang_code = code

    def set_flavor(self, flavor: Optional[str] = None):
        if flavor is None:
            flavor = self.list_flavors()[0]
        if flavor == self._flavor:
            return
        self.unload_model()

        self._flavor = flavor
        _lang_code = self.lang_code if self.lang_code != "kr" else "jp"

        dir_model = self.working_dir + f".s1_{_lang_code}"
        repo_id = self.repo_id[f"s1_{_lang_code}"]
        if not os.path.exists(os.path.join(dir_model, "onnx", f"model_{flavor}.onnx")):
            download_files(repo_id, "onnx", dir_model, f"model_{flavor}.*")
            download_files(repo_id, ".", dir_model, "*.json")

        with open(os.path.join(dir_model, "config.json"), "r", encoding="utf-8") as reader:
            self.config_s1 = PretrainedConfig(**json.load(reader))

        dir_model = self.working_dir + ".s2"
        if not os.path.exists(os.path.join(dir_model, "onnx", f"model_{flavor}.onnx")):
            download_files(self.repo_id["s2"], "onnx", dir_model, f"model_{flavor}.*")
            download_files(self.repo_id["s2"], ".", dir_model, "*.json")

        with open(os.path.join(dir_model, "config.json"), "r", encoding="utf-8") as reader:
            self.config_s2 = PretrainedConfig(**json.load(reader))

        # _codec_dir = os.path.dirname(self.codec_dir)
        # check_fn = os.path.join(_codec_dir, ".done")
        # if not os.path.exists(check_fn):
        #     download_files(self.codec_id, "onnx", _codec_dir, "decoder_model.onnx")
        #     with open(check_fn, "w") as writer:
        #         writer.write("\n")

    def gen_args(self):
        self._keys = ["lyrics", "audio_prompt", "segments", "duration", "penalty"]
        self._types = ["text_area", "toogle", "slider", "slider", "slider"]
        self._labels = [
            "Lyrics",
            "Use Audio Prompt",
            "Number of Segments",
            "Duration of song",
            "Repetition penalty",
        ]
        self._values = [None, False, 2, 15, 1.1]
        self._kwargs = [
            {"height": 200},
            {},
            {"min_value": 1, "max_value": 5, "step": 1},
            {"min_value": 1, "max_value": 45, "step": 1},
            {"min_value": 0.5, "max_value": 1.5, "step": 0.1}
        ]

    def load_model(self):
        if self._loaded:
            return
        if self._flavor is None:
            self.set_flavor()
        use_gpu = "CUDAExecutionProvider" in ort.get_available_providers()
        providers = [
            ("CUDAExecutionProvider", {"device_id": self.num_gpus - 1})
        ] if use_gpu else []
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        # sess_options.log_severity_level=1
        # print(providers)
        _lang_code = self.lang_code if self.lang_code != "kr" else "jp"
        model_path = os.path.join(
            self.working_dir + f".s1_{_lang_code}", "onnx", f"model_{self._flavor}.onnx"
        )
        self.model = {}
        model = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.model["s1"] = ORTModelForCausalLM(
            model,
            self.config_s1,
            use_io_binding=True,
            use_cache=True,
        )

        model_path = os.path.join(
            self.working_dir + ".s2", "onnx", f"model_{self._flavor}.onnx"
        )
        model = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.model["s2"] = ORTModelForCausalLM(
            model,
            self.config_s2,
            use_io_binding=True,
            use_cache=True,
        )

        self._loaded = True
        
        self.mmtokenizer = MMSentencePieceTokenizer(os.path.join(
            self.working_dir + f".s1_{_lang_code}", "tokenizer.model")
        )

    def _format_promt(self, prompt, voice="zoe"):
        adapated_prompt = f"{voice}: {prompt}"
        tokens = self.tokenizer(adapated_prompt, return_tensors="np").input_ids[0]
        tokens = np.concat([[128259], tokens, [128009, 128260]], axis=0)[None]
        tokens = torch.from_numpy(tokens).to(device="cuda")
        return tokens, torch.ones_like(tokens)

    def _generate_wav(self, tokens):
        parsed_tkns = parse_output(tokens)
        codes = redistribute_codes(parsed_tkns)
        audio: np.ndarray = self.codec.run(None, codes)[0]
        return audio.squeeze()
        
    def _generate(
        self,
        text: str,
        lyrics: str,
        progress_callback=None,
        progress_state=None,
        progress=None,
        audio_prompt = None,
        segments: int = 1,
        duration: int = 10,
        penalty: float = 1.1,
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

        self.load_model()
        run_n_segments = 2
        max_new_tokens = 45
        max_new_tokens = int(max_new_tokens * (100/run_n_segments))
        
        genres = text.strip()
        lyrics = split_lyrics(lyrics + "\n")
        full_lyrics = "\n".join(lyrics)
        prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
        prompt_texts += lyrics
        
        top_p = 0.93
        temperature = 1.0
        repetition_penalty = 1.2
        start_of_segment = self.mmtokenizer.tokenize('[start_of_segment]')
        end_of_segment = self.mmtokenizer.tokenize('[end_of_segment]')

        print(start_of_segment)
        # Initialize tracking
        # text = text.replace("\n", " ")
        # # Process by utterance instead of paragraph
        # utters = [f"{x.lstrip()}." for x in text.split(".") if len(x) > 1]

        # audio = []
        # n_utters = float(len(utters))
        # for idx, line in enumerate(utters):
        #     input_ids, attention_mask = self._format_promt(line, voice_name)
        #     # print(prompt)
        #     output = self.model.generate(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         max_new_tokens=max_new_tokens,
        #         do_sample=True,
        #         temperature=temperature,
        #         top_p=top_p,
        #         repetition_penalty=repetition_penalty,
        #         num_return_sequences=1,
        #         eos_token_id=128258,
        #     )
        #     segment = self._generate_wav(output)
        #     audio.append(segment)
        #     if progress is not None:
        #         progress((idx + 1) / n_utters)
        # audio = np.concat(audio, axis=0)
        # return audio


my_text = """inspiring female uplifting pop airy vocal electronic bright vocal vocal"""
my_lyrics = """
[verse]
Staring at the sunset, colors paint the sky
Thoughts of you keep swirling, can't deny
I know I let you down, I made mistakes
But I'm here to mend the heart I didn't break

[chorus]
Every road you take, I'll be one step behind
Every dream you chase, I'm reaching for the light
You can't fight this feeling now
I won't back down
You know you can't deny it now
I won't back down

[verse]
They might say I'm foolish, chasing after you
But they don't feel this love the way we do
My heart beats only for you, can't you see?
I won't let you slip away from me
"""


def _main():
    tts_model = YueTTSONNX()
    tts_model.initialize()
    print(tts_model.list_flavors())
    print(tts_model.list_languages())
    tts_model.set_flavor()
    output = tts_model.generate(my_text, lyrics=my_lyrics)
    # sf.write("test_yue.wav", output, tts_model.samplerate)


if __name__ == "__main__":
    _main()
