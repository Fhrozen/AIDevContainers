import glob
import json
import os
from typing import List, Optional, Tuple

import torch
import numpy as np
import onnxruntime as ort
from utils import download_files
from transformers import AutoTokenizer, PretrainedConfig
import soundfile as sf

from optimum.onnxruntime import ORTModelForCausalLM


LANG_CODES = {
    'en-us': 'American English',
}


def parse_output(generated_ids):
    token_to_find = 128257
    token_to_remove = 128258
    
    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
    else:
        cropped_tensor = generated_ids

    processed_rows = []
    for row in cropped_tensor:
        masked_row = row[row != token_to_remove]
        processed_rows.append(masked_row)

    code_lists = []
    for row in processed_rows:
        row_length = row.size(0)
        new_length = (row_length // 7) * 7
        trimmed_row = row[:new_length]
        trimmed_row = [t - 128266 for t in trimmed_row]
        code_lists.append(trimmed_row)
    return torch.tensor(code_lists[0]).cpu().numpy()  # Return just the first one for single sample


# Redistribute codes for audio generation
def redistribute_codes(code_list):
    layer_1 = []
    layer_2 = []
    layer_3 = []
    for i in range((len(code_list)+1)//7):
        layer_1.append(code_list[7*i])
        layer_2.append(code_list[7*i+1]-4096)
        layer_3.append(code_list[7*i+2]-(2*4096))
        layer_3.append(code_list[7*i+3]-(3*4096))
        layer_2.append(code_list[7*i+4]-(4*4096))
        layer_3.append(code_list[7*i+5]-(5*4096))
        layer_3.append(code_list[7*i+6]-(6*4096))
        
    layer_1 = np.array(layer_1)[None]
    layer_2 = np.array(layer_2)[None]
    layer_3 = np.array(layer_3)[None]
    return {
        "audio_codes.0": layer_1,
        "audio_codes.1": layer_2,
        "audio_codes.2": layer_3,
    }


class OrpheusTTSModelONNX:
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(__file__), "downloaded_orpheus", "onnx")
        self.codec_dir = os.path.join(os.path.dirname(__file__), "downloaded_orpheus/codec", "onnx")
        self.repo_id = "onnx-community/orpheus-3b-0.1-ft-ONNX"
        self.codec_id = "onnx-community/snac_24khz-ONNX"
        _vars = {
            "using_voice": None,
            "model": None,
            "lang_code": None,
            "tokenizer": None,
            "samplerate": 24000,
            "flavor": None,
            "_loaded": False,
        }
        _ = [setattr(self, k, v) for k, v in _vars.items()]

    def initialize(self) -> bool:
        """Initialize"""
        try:
            print("Initializing model...")
            self.set_flavor()
            self.set_language()
            self.set_voice()
            print("Model initialization complete")
            return True
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return False

    def list_flavors(self) -> List[str]:
        """List available models from voices_v1 directory"""
        _models_dir = os.path.dirname(self.models_dir)
        check_fn = os.path.join(_models_dir, ".done")
        if not os.path.exists(check_fn):
            download_files(self.repo_id, "onnx", _models_dir, "model_q*")
            download_files(self.repo_id, ".", _models_dir, "*.json")
            with open(check_fn, "w") as writer:
                writer.write("\n")
        
        _codec_dir = os.path.dirname(self.codec_dir)
        check_fn = os.path.join(_codec_dir, ".done")
        if not os.path.exists(check_fn):
            download_files(self.codec_id, "onnx", _codec_dir, "decoder_model.onnx")
            with open(check_fn, "w") as writer:
                writer.write("\n")

        models = []
        if os.path.exists(self.models_dir):
            listfiles = glob.glob(os.path.join(self.models_dir, "*.onnx"))
            for filename in listfiles:
                model_name = filename.split(".onnx")[0]
                models.append(os.path.basename(model_name))
        return models

    def list_properties(self):
        return

    def list_languages(self) -> List[str]:
        """List available voices from voices_v1 directory"""
        langs = ["English"]
        return langs

    def list_voices(self) -> List[str]:
        """List available voices from voices_v1 directory"""
        _voices_dir = os.path.dirname(self.models_dir)
        voices = ["zoe", "zac","jess", "leo", "mia", "julia", "leah"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            _voices_dir
        )

        with open(os.path.join(_voices_dir, "config.json"), "r", encoding="utf-8") as reader:
            self.config = PretrainedConfig(**json.load(reader))
        return sorted(voices)

    def set_language(self, langid: Optional[str] = None):
        pass

    def set_flavor(self, flavor: Optional[str] = None):
        if flavor is None:
            flavor = self.list_flavors()[0]
        if flavor != self.flavor:
            self.unload_model()
        self.flavor = flavor

    def set_voice(self, voice: Optional[str] = None):
        if voice is None:
            voice = self.list_voices()[0]
        self.voice = voice

    def load_model(self):
        if self._loaded:
            return
        if self.flavor is None:
            self.set_flavor()
        use_gpu = "CUDAExecutionProvider" in ort.get_available_providers()
        providers = ["CUDAExecutionProvider"] if use_gpu else []
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        # sess_options.log_severity_level=1
        # print(providers)
        model_path = os.path.join(self.models_dir, f"{self.flavor}.onnx")
        model = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.model = ORTModelForCausalLM(
            model,
            self.config,
            use_io_binding=True,
            use_cache=True,
        )
        
        codec_path = os.path.join(self.codec_dir, "decoder_model.onnx")
        self.codec = ort.InferenceSession(codec_path, sess_options, providers=providers)
        self._loaded = True

    def unload_model(self):
        self._loaded = False
        self.model = None

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
        
    def generate(
        self,
        text: str,
        voice_names: list[str] | str,
        speed: float = 1.0,
        progress_callback=None,
        progress_state=None,
        progress=None,
        max_new_tokens: int = 1200,
        temperature: float = 0.6,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
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
        if not text or not voice_names:
            raise ValueError("Text and voice name are required")
        self.load_model()
        # Handle voice selection
        voice_name = voice_names[0] if isinstance(voice_names, list) else voice_names

        # Initialize tracking
        text = text.replace("\n", " ")
        # Process by utterance instead of paragraph
        utters = [f"{x.lstrip()}." for x in text.split(".") if len(x) > 1]

        audio = []
        n_utters = float(len(utters))
        for idx, line in enumerate(utters):
            input_ids, attention_mask = self._format_promt(line, voice_name)
            # print(prompt)
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                eos_token_id=128258,
            )
            segment = self._generate_wav(output)
            audio.append(segment)
            if progress is not None:
                progress((idx + 1) / n_utters)
        audio = np.concat(audio, axis=0)
        return audio


my_text = """His eye fell on the yellow book that Lord Henry had sent him. What was
it, he wondered. He went towards the little, pearl-coloured octagonal
stand that had always looked to him like the work of some strange
Egyptian bees that wrought in silver, and taking up the volume, flung
himself into an arm-chair and began to turn over the leaves. After a
few minutes he became absorbed. It was the strangest book that he had
ever read. It seemed to him that in exquisite raiment, and to the
delicate sound of flutes, the sins of the world were passing in dumb
show before him. Things that he had dimly dreamed of were suddenly made
real to him. Things of which he had never dreamed were gradually
revealed."""



if __name__ == "__main__":
    tts_model = OrpheusTTSModelONNX()
    tts_model.initialize()
    voices = tts_model.list_voices()
    print(tts_model.list_voices())
    print(tts_model.list_flavors())
    print(tts_model.list_languages())
    output = tts_model.generate(my_text, "zoe")
    sf.write("test_orpheus.wav", output, tts_model.samplerate)

    # tts_model.set_language("British English")
    # tts_model.set_language("Japanese")
    # tts_model.set_language("Spanish")
    # tts_model.generate(my_text, ["jf_alpha"])
    # tts_model.generate(my_text, ["jf_alpha"])
    # for lang in LANG_CODES.values():
    #     print(f"setting lang {lang}")
    #     tts_model.set_language(lang)
    #     time.sleep(2)