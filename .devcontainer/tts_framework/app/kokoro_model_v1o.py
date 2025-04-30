import os
from typing import Generator, List, Optional, Tuple
import time
import re
import glob

import logging

import librosa
from misaki import en, espeak, ja, zh
import numpy as np
from huggingface_hub import (
    hf_hub_download,
    snapshot_download
)

import onnxruntime
from onnxruntime import InferenceSession


LANG_CODES = {
    'en-us': 'American English',
    'en-gb': 'British English',
    'es': 'Spanish',
    'fr-fr': 'French',
    'hi': 'Hindi',
    'it': 'Italian',
    'pt-br': 'Portuguese (Br)',
    'ja': 'Japanese',
    'zh': 'Mandarin Chinese',
}

def get_vocab():
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»“” '
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    dicts = {}
    for i in range(len((symbols))):
        dicts[symbols[i]] = i
    return dicts


def download_files(repo_id: str, directory: str, local_dir: str) -> None:
    """Download voice files from Hugging Face Hub
    
    Args:
        repo_id: The Hugging Face repository ID
        directory: The directory in the repo to download (e.g. "voices")
        local_dir: Local directory to save files to
    """
    os.makedirs(local_dir, exist_ok=True)
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


class KokoroModelV1ONNX:
    """TTS model for v1.0.0-onnx"""
    VOCAB = get_vocab()
    def __init__(self):
        self.g2p = None
        self.voices_dir = os.path.join(os.path.dirname(__file__), "voices_v1o", "voices")
        self.models_dir = os.path.join(os.path.dirname(__file__), "models_v1o", "onnx")
        self.repo_id = "onnx-community/Kokoro-82M-v1.0-ONNX"
        self.using_voice = None
        self.model = None
        self.lang_code = None
        self.tokenize = None

    def initialize(self) -> bool:
        """Initialize"""
        try:
            print("Initializing v1.0.0 model...")
            self.g2p = None
            self.set_model()
            self.set_language()
            self.set_voice()
            print("Model initialization complete")
            return True
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return False

    def list_voices(self) -> List[str]:
        """List available voices from voices_v1 directory"""
        _voices_dir = os.path.dirname(self.voices_dir)
        check_fn = os.path.join(_voices_dir, ".done")
        if not os.path.exists(check_fn):
            download_files(self.repo_id, "voices", _voices_dir)
            with open(check_fn, "w") as writer:
                writer.write("\n")

        voices = []
        if os.path.exists(self.voices_dir):
            listfiles = glob.glob(os.path.join(self.voices_dir, "*.bin"))
            for filename in listfiles:
                voice_name = filename.split(".bin")[0]
                voices.append(os.path.basename(voice_name))
        return sorted(voices)

    def list_languages(self) -> List[str]:
        """List available voices from voices_v1 directory"""
        langs = list(LANG_CODES.values())
        return langs

    def list_flavors(self) -> List[str]:
        """List available models from voices_v1 directory"""
        _models_dir = os.path.dirname(self.models_dir)
        check_fn = os.path.join(_models_dir, ".done")
        if not os.path.exists(check_fn):
            download_files(self.repo_id, "onnx", _models_dir)
            with open(check_fn, "w") as writer:
                writer.write("\n")

        models = []
        if os.path.exists(self.models_dir):
            listfiles = glob.glob(os.path.join(self.models_dir, "*.onnx"))
            for filename in listfiles:
                model_name = filename.split(".onnx")[0]
                models.append(os.path.basename(model_name))
        return models

    def set_language(self, langid: Optional[str] = None):
        if langid is None:
            code = "en-us"
        else:
            _codes = {v:k for k, v in LANG_CODES.items()}
            code = _codes[langid]

        self.tokenize = self.extra_tokenize
        if code.startswith("en-"):
            self.tokenize = self.en_tokenize
            try:
                fallback = espeak.EspeakFallback(british=code=='en-gb')
            except Exception as e:
                logging.warning("EspeakFallback not Enabled: OOD words will be skipped")
                logging.warning({str(e)})
                fallback = None
            self.g2p = en.G2P(trf=False, british=code=='en-gb', fallback=fallback, unk='')
        elif code == "ja":
            self.g2p = ja.JAG2P()
        elif code == "zh":
            self.g2p = zh.ZHG2P()
        else:
            self.g2p = espeak.EspeakG2P(language=code)

        self.lang_code = code

    def set_model(self, model_name: Optional[str] = None):
        if model_name is None:
            model_name = self.list_flavors()[0]

        use_gpu = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        providers = ["CUDAExecutionProvider"] if use_gpu else []
        providers.append("CPUExecutionProvider")

        sess_options = onnxruntime.SessionOptions()
        # sess_options.log_severity_level=1
        # print(providers)
        model_path = os.path.join(self.models_dir, f"{model_name}.onnx")
        self.model = InferenceSession(model_path, sess_options, providers=providers)
        # print(self.model)

    def set_voice(self, voice: Optional[str] = None):
        if voice is None:
            voice = self.list_voices()[0]
        self.voice = np.fromfile(
            os.path.join(self.voices_dir, f"{voice}.bin"),
            dtype=np.float32
        ).reshape(-1, 1, 256)

    @classmethod
    def waterfall_last(
        cls,
        tokens: List[en.MToken],
        next_count: int,
        waterfall: List[str] = ['!.?…', ':;', ',—'],
        bumps: List[str] = [')', '”']
    ) -> int:
        for w in waterfall:
            z = next((i for i, t in reversed(list(enumerate(tokens))) if t.phonemes in set(w)), None)
            if z is None:
                continue
            z += 1
            if z < len(tokens) and tokens[z].phonemes in bumps:
                z += 1
            if next_count - len(cls.tokens_to_ps(tokens[:z])) <= 510:
                return z
        return len(tokens)

    @classmethod
    def tokens_to_text(cls, tokens: List[en.MToken]) -> str:
        return ''.join(t.text + t.whitespace for t in tokens).strip()

    @classmethod
    def tokens_to_ps(cls, tokens: List[en.MToken]) -> str:
        return ''.join(t.phonemes + (' ' if t.whitespace else '') for t in tokens).strip()

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

    def en_tokenize(
        self,
        tokens: List[en.MToken]
    ) -> Generator[Tuple[str, str, List[en.MToken]], None, None]:
        tks = []
        pcount = 0
        for t in tokens:
            # American English: ɾ => T
            t.phonemes = '' if t.phonemes is None else t.phonemes.replace('ɾ', 'T')
            next_ps = t.phonemes + (' ' if t.whitespace else '')
            next_pcount = pcount + len(next_ps.rstrip())
            if next_pcount > 510:
                z = self.waterfall_last(tks, next_pcount)
                text = self.tokens_to_text(tks[:z])
                logging.debug(f"Chunking text at {z}: '{text[:30]}{'...' if len(text) > 30 else ''}'")
                ps = self.tokens_to_ps(tks[:z])
                yield text, ps, tks[:z]
                tks = tks[z:]
                pcount = len(self.tokens_to_ps(tks))
                if not tks:
                    next_ps = next_ps.lstrip()
            tks.append(t)
            pcount += len(next_ps)
        if tks:
            text = self.tokens_to_text(tks)
            ps = self.tokens_to_ps(tks)
            yield ''.join(text).strip(), ''.join(ps).strip(), tks

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

    def generate_speech(
        self,
        text: str,
        voice_names: list[str],
        speed: float = 1.0,
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
        if not text or not voice_names:
            raise ValueError("Text and voice name are required")

         # Handle voice selection
        voice_name = voice_names[0]

        # Initialize tracking
        audio_chunks = []
        chunk_times = []
        chunk_sizes = []
        total_tokens = 0

        try:
            start_time = time.time()

            # Preprocess text - replace single newlines with spaces while preserving paragraphs
            processed_text = '\n\n'.join(
                paragraph.replace('\n', ' ').replace('  ', ' ').strip()
                for paragraph in text.split('\n\n')
            )
            self.set_voice(voice_name)

            if isinstance(processed_text, str):
                processed_text = re.split(split_pattern, processed_text.strip()) if split_pattern else [processed_text]

            # Process chunks
            total_duration = 0  # Total audio duration in seconds
            total_process_time = 0  # Total processing time in seconds
            chunk = 0

            for graphemes in processed_text:
                tokens = self.g2p(graphemes)
                if self.lang_code.startswith("en-"):
                    tokens = tokens[1]

                for gs, ps, _ in self.tokenize(tokens):
                    if not ps:
                        continue
                    elif len(ps) > 510:
                        logging.warning(f"Unexpected len(ps) == {len(ps)} > 510 and ps == '{ps}'")
                        ps = ps[:510]

                    audio_part = self.infer(ps, speed)
                    if trim:
                        # Trim leading and trailing silence for a more natural sound concatenation
                        # (initial ~2s, subsequent ~0.02s)
                        audio_part, _ = librosa.effects.trim(audio_part)  

                    chunk_process_time = time.time() - start_time - total_process_time
                    total_process_time += chunk_process_time
                    audio_chunks.append(audio_part)

                    # Calculate metrics
                    chunk_tokens = len(gs)
                    total_tokens += chunk_tokens

                    # Calculate audio duration
                    chunk_duration = len(audio_part) / 24000  # Convert samples to seconds
                    total_duration += chunk_duration

                    # Calculate speed metrics
                    tokens_per_sec = chunk_tokens / chunk_duration  # Tokens per second of audio
                    rtf = chunk_process_time / chunk_duration  # Real-time factor

                    chunk_times.append(chunk_process_time)
                    chunk_sizes.append(chunk_tokens)

                    print(f"Chunk {chunk + 1}:")
                    print(f"  Process time: {chunk_process_time:.2f}s")
                    print(f"  Audio duration: {chunk_duration:.2f}s")
                    print(f"  Tokens/sec: {tokens_per_sec:.1f}")
                    print(f"  Real-time factor: {rtf:.3f}")
                    print(f"  Speed: {(1/rtf):.1f}x real-time")
                    chunk += 1
                
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
                            chunk + 1,
                            -1,  # Let UI handle total chunks
                            tokens_per_sec,
                            rtf,
                            progress_state,
                            start_time,
                            gpu_timeout,
                            progress
                        )

                audio_chunks.append(np.zeros([12000], dtype=np.float32)) # append 0.5 secs after a paragraph
            # # Concatenate audio chunks
            audio = np.concatenate(audio_chunks)

            # # Return audio and metrics
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
            logging.error(f"Error generating speech: {str(e)}")
            raise


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

# my_text = """Don Quijote de la Mancha es una novela escrita por el español Miguel de Cervantes Saavedra.
# Publicada su primera parte con el título de El ingenioso hidalgo don Quijote de la Mancha a comienzos de 1605,
# es la obra más destacada de la literatura española y una de las principales de la literatura universal.
# En 1615 apareció su continuación con el título de Segunda parte del
# ingenioso caballero don Quijote de la Mancha.
# El Quijote de 1605 se publicó dividido en cuatro partes; pero al aparecer el Quijote de 1615 en
# calidad de Segunda parte de la obra, quedó revocada de hecho la partición en cuatro secciones
# del volumen publicado diez años antes por Cervantes."""


# my_text = """司祭と床屋がドン・キホーテの蔵書を批評する場面ではセルバンテス自身の作品である『ラ・ガラテーア』
# も取り上げられている. 後編では「前編が出版されて世に出回っている」という設定となっており、登場人物たちが前編の批評を行い、
# 矛盾している記述の釈明を行ったりする.また、前編でドン・キホーテを知った人々が、
# 前編での記述をもとにドン・キホーテ主従に悪戯をしかける. また、上述の贋作についても度々言及されており、「ドン・キホーテを騙る人物が存在し、贋作はこの2人の道中記である」
# という設定になっている."""


if __name__ == "__main__":
    tts_model = TTSModelV1ONNX()
    tts_model.initialize()
    voices = tts_model.list_voices()
    print(tts_model.list_voices())
    print(tts_model.list_flavors())
    print(tts_model.list_languages())

    tts_model.set_language("British English")
    # tts_model.set_language("Japanese")
    # tts_model.set_language("Spanish")
    tts_model.generate_speech(my_text, ["jf_alpha"])
    # for lang in LANG_CODES.values():
    #     print(f"setting lang {lang}")
    #     tts_model.set_language(lang)
    #     time.sleep(2)