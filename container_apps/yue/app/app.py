import gradio as gr
# import subprocess
import os
# import shutil
import tempfile
import torch
import sys
import uuid
import re

from huggingface_hub import snapshot_download

# Create xcodec_mini_infer folder
folder_path = './xcodec_mini_infer'

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.mkdir(folder_path)
    print(f"Folder created at: {folder_path}")
else:
    print(f"Folder already exists at: {folder_path}")

snapshot_download(
    repo_id="m-a-p/xcodec_mini_infer",
    local_dir="./xcodec_mini_infer"
)

# Change to the "inference" directory
inference_dir = "."
try:
    os.chdir(inference_dir)
    print(f"Changed working directory to: {os.getcwd()}")
except FileNotFoundError:
    print(f"Directory not found: {inference_dir}")
    exit(1)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer', 'descriptaudiocodec'))

# don't change above code

import argparse
import numpy as np
import json
from omegaconf import OmegaConf
import torchaudio
from torchaudio.transforms import Resample
import soundfile as sf

from tqdm import tqdm
from einops import rearrange
from codecmanipulator import CodecManipulator
from mmtokenizer import _MMSentencePieceTokenizer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList
)
# import glob
# import time
# import copy
# from collections import Counter
# from models.soundstream_hubert_new import SoundStream
#from vocoder import build_codec_model, process_audio # removed vocoder
#from post_process_audio import replace_low_freq_with_energy_matched # removed post process

device = "cuda:0"
# device = "cpu"

# model_tag = "tensorblock/YuE-s1-7B-anneal-en-cot-GGUF"
# gguf_file = "YuE-s1-7B-anneal-en-cot-Q4_K_M.gguf"
model_tag = "m-a-p/YuE-s1-7B-anneal-en-cot"
gguf_file = None
kwargs = {
    "quantization_config": BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_storage="uint8",
        bnb_4bit_quant_type="fp4",
        bnb_4bit_use_double_quant=False,
    )
}
model = AutoModelForCausalLM.from_pretrained(
    model_tag,
    torch_dtype=torch.float16,
    gguf_file=gguf_file,
    attn_implementation="flash_attention_2",
    **kwargs,
    # low_cpu_mem_usage=True,
).to(device)
model.eval()

basic_model_config = './xcodec_mini_infer/final_ckpt/config.yaml'
resume_path = './xcodec_mini_infer/final_ckpt/ckpt_00360000.pth'
#config_path = './xcodec_mini_infer/decoders/config.yaml' # removed vocoder
#vocal_decoder_path = './xcodec_mini_infer/decoders/decoder_131000.pth' # removed vocoder
#inst_decoder_path = './xcodec_mini_infer/decoders/decoder_151000.pth' # removed vocoder

mmtokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")

codectool = CodecManipulator("xcodec", 0, 1)
model_config = OmegaConf.load(basic_model_config)
# Load codec model
codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)
parameter_dict = torch.load(resume_path, map_location='cpu', weights_only=False)
codec_model.load_state_dict(parameter_dict['codec_model'])
# codec_model = torch.compile(codec_model)
codec_model.eval()

# Preload and compile vocoders # removed vocoder
#vocal_decoder, inst_decoder = build_codec_model(config_path, vocal_decoder_path, inst_decoder_path)
#vocal_decoder.to(device)
#inst_decoder.to(device)
#vocal_decoder = torch.compile(vocal_decoder)
#inst_decoder = torch.compile(inst_decoder)
#vocal_decoder.eval()
#inst_decoder.eval()


def generate_music(
        max_new_tokens=5,
        run_n_segments=2,
        genre_txt=None,
        lyrics_txt=None,
        use_audio_prompt=False,
        audio_prompt_path="",
        prompt_start_time=0.0,
        prompt_end_time=30.0,
        cuda_idx=0,
        rescale=False,
):
    if use_audio_prompt and not audio_prompt_path:
        raise FileNotFoundError("Please offer audio prompt filepath using '--audio_prompt_path', when you enable 'use_audio_prompt'!")
    cuda_idx = cuda_idx
    max_new_tokens = max_new_tokens * 100

    with tempfile.TemporaryDirectory() as output_dir:
        stage1_output_dir = os.path.join(output_dir, f"stage1")
        os.makedirs(stage1_output_dir, exist_ok=True)

        class BlockTokenRangeProcessor(LogitsProcessor):
            def __init__(self, start_id, end_id):
                self.blocked_token_ids = list(range(start_id, end_id))

            def __call__(self, input_ids, scores):
                scores[:, self.blocked_token_ids] = -float("inf")
                return scores

        def load_audio_mono(filepath, sampling_rate=16000):
            audio, sr = torchaudio.load(filepath)
            # Convert to mono
            audio = torch.mean(audio, dim=0, keepdim=True)
            # Resample if needed
            if sr != sampling_rate:
                resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
                audio = resampler(audio)
            return audio

        def split_lyrics(lyrics: str):
            pattern = r"\[(\w+)\](.*?)\n(?=\[|\Z)"
            segments = re.findall(pattern, lyrics, re.DOTALL)
            structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
            return structured_lyrics

        # Call the function and print the result
        stage1_output_set = []

        genres = genre_txt.strip()
        lyrics = split_lyrics(lyrics_txt + "\n")
        # intruction
        full_lyrics = "\n".join(lyrics)
        prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
        prompt_texts += lyrics

        random_id = uuid.uuid4()
        output_seq = None
        # Here is suggested decoding config
        top_p = 0.93
        temperature = 1.0
        repetition_penalty = 1.2
        # special tokens
        start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
        end_of_segment = mmtokenizer.tokenize('[end_of_segment]')

        raw_output = None

        # Format text prompt
        run_n_segments = min(run_n_segments + 1, len(lyrics))

        print(list(enumerate(tqdm(prompt_texts[:run_n_segments]))))

        for i, p in enumerate(tqdm(prompt_texts[:run_n_segments])):
            section_text = p.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
            guidance_scale = 1.5 if i <= 1 else 1.2
            if i == 0:
                continue
            if i == 1:
                if use_audio_prompt:
                    audio_prompt = load_audio_mono(audio_prompt_path)
                    audio_prompt.unsqueeze_(0)
                    with torch.no_grad():
                        raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=0.5)
                    raw_codes = raw_codes.transpose(0, 1)
                    raw_codes = raw_codes.cpu().numpy().astype(np.int16)
                    # Format audio prompt
                    code_ids = codectool.npy2ids(raw_codes[0])
                    audio_prompt_codec = code_ids[int(prompt_start_time * 50): int(prompt_end_time * 50)]  # 50 is tps of xcodec
                    audio_prompt_codec_ids = [mmtokenizer.soa] + codectool.sep_ids + audio_prompt_codec + [
                        mmtokenizer.eoa]
                    sentence_ids = mmtokenizer.tokenize("[start_of_reference]") + audio_prompt_codec_ids + mmtokenizer.tokenize(
                        "[end_of_reference]")
                    head_id = mmtokenizer.tokenize(prompt_texts[0]) + sentence_ids
                else:
                    head_id = mmtokenizer.tokenize(prompt_texts[0])
                prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids
            else:
                prompt_ids = end_of_segment + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids

            prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device)
            input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 1 else prompt_ids
            # Use window slicing in case output sequence exceeds the context of model
            max_context = 16384 - max_new_tokens - 1
            if input_ids.shape[-1] > max_context:
                print(
                    f'Section {i}: output length {input_ids.shape[-1]} exceeding context length {max_context}, now using the last {max_context} tokens.')
                input_ids = input_ids[:, -(max_context):]
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
                output_seq = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=100,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=mmtokenizer.eoa,
                    pad_token_id=mmtokenizer.eoa,
                    logits_processor=LogitsProcessorList([BlockTokenRangeProcessor(0, 32002), BlockTokenRangeProcessor(32016, 32016)]),
                    guidance_scale=guidance_scale,
                    use_cache=True
                )
                if output_seq[0][-1].item() != mmtokenizer.eoa:
                    tensor_eoa = torch.as_tensor([[mmtokenizer.eoa]]).to(model.device)
                    output_seq = torch.cat((output_seq, tensor_eoa), dim=1)
            if i > 1:
                raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1]:]], dim=1)
            else:
                raw_output = output_seq
            print(len(raw_output))

        # save raw output and check sanity
        ids = raw_output[0].cpu().numpy()
        soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
        eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()
        if len(soa_idx) != len(eoa_idx):
            raise ValueError(f'invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}')

        vocals = []
        instrumentals = []
        range_begin = 1 if use_audio_prompt else 0
        for i in range(range_begin, len(soa_idx)):
            codec_ids = ids[soa_idx[i] + 1:eoa_idx[i]]
            if codec_ids[0] == 32016:
                codec_ids = codec_ids[1:]
            codec_ids = codec_ids[:2 * (codec_ids.shape[0] // 2)]
            vocals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
            vocals.append(vocals_ids)
            instrumentals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])
            instrumentals.append(instrumentals_ids)
        vocals = np.concatenate(vocals, axis=1)
        instrumentals = np.concatenate(instrumentals, axis=1)

        vocal_save_path = os.path.join(stage1_output_dir, f"vocal_{random_id}".replace('.', '@') + '.npy')
        inst_save_path = os.path.join(stage1_output_dir, f"instrumental_{random_id}".replace('.', '@') + '.npy')
        np.save(vocal_save_path, vocals)
        np.save(inst_save_path, instrumentals)
        stage1_output_set.append(vocal_save_path)
        stage1_output_set.append(inst_save_path)

        print("Converting to Audio...")

        # convert audio tokens to audio
        def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
            folder_path = os.path.dirname(path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            limit = 0.99
            max_val = wav.abs().max()
            wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
            torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

        # reconstruct tracks
        recons_output_dir = os.path.join(output_dir, "recons")
        recons_mix_dir = os.path.join(recons_output_dir, 'mix')
        os.makedirs(recons_mix_dir, exist_ok=True)
        tracks = []
        for npy in stage1_output_set:
            codec_result = np.load(npy)
            decodec_rlt = []
            with torch.no_grad():
                decoded_waveform = codec_model.decode(
                    torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to(
                        device))
            decoded_waveform = decoded_waveform.cpu().squeeze(0)
            decodec_rlt.append(torch.as_tensor(decoded_waveform))
            decodec_rlt = torch.cat(decodec_rlt, dim=-1)
            save_path = os.path.join(recons_output_dir, os.path.splitext(os.path.basename(npy))[0] + ".mp3")
            tracks.append(save_path)
            save_audio(decodec_rlt, save_path, 16000)
        # mix tracks
        for inst_path in tracks:
            try:
                if (inst_path.endswith('.wav') or inst_path.endswith('.mp3')) \
                        and 'instrumental' in inst_path:
                    # find pair
                    vocal_path = inst_path.replace('instrumental', 'vocal')
                    if not os.path.exists(vocal_path):
                        continue
                    # mix
                    recons_mix = os.path.join(recons_mix_dir, os.path.basename(inst_path).replace('instrumental', 'mixed'))
                    vocal_stem, sr = sf.read(inst_path)
                    instrumental_stem, _ = sf.read(vocal_path)
                    mix_stem = (vocal_stem + instrumental_stem) / 1
                    return (sr, (mix_stem * 32767).astype(np.int16)), (sr, (vocal_stem * 32767).astype(np.int16)), (sr, (instrumental_stem * 32767).astype(np.int16))
            except Exception as e:
                print(e)
                return None, None, None


def infer(genre_txt_content, lyrics_txt_content, num_segments=2, max_new_tokens=15):
    # Execute the command
    try:
        mixed_audio_data, vocal_audio_data, instrumental_audio_data = generate_music(genre_txt=genre_txt_content, lyrics_txt=lyrics_txt_content, run_n_segments=num_segments,
                               cuda_idx=0, max_new_tokens=max_new_tokens)
        return mixed_audio_data, vocal_audio_data, instrumental_audio_data
    except Exception as e:
        gr.Warning("An Error Occured: " + str(e))
        return None, None, None
    finally:
        print("Temporary files deleted.")


# Gradio
with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("# YuE: Open Music Foundation Models for Full-Song Generation")
        gr.HTML("""
        <div style="display:flex;column-gap:4px;">
            <a href="https://github.com/multimodal-art-projection/YuE">
                <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
            </a>
            <a href="https://map-yue.github.io">
                <img src='https://img.shields.io/badge/Project-Page-green'>
            </a>
            <a href="https://huggingface.co/spaces/innova-ai/YuE-music-generator-demo?duplicate=true">
                <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-sm.svg" alt="Duplicate this Space">
            </a>
        </div>
        """)
        with gr.Row():
            with gr.Column():
                genre_txt = gr.Textbox(label="Genre")
                lyrics_txt = gr.Textbox(label="Lyrics")

            with gr.Column():
                num_segments = gr.Number(label="Number of Segments", value=2, interactive=True)
                max_new_tokens = gr.Slider(label="Duration of song", minimum=1, maximum=30, step=1, value=15, interactive=True)
                submit_btn = gr.Button("Submit")

                music_out = gr.Audio(label="Mixed Audio Result")
                with gr.Accordion(label="Vocal and Instrumental Result", open=False):
                    vocal_out = gr.Audio(label="Vocal Audio")
                    instrumental_out = gr.Audio(label="Instrumental Audio")

        gr.Examples(
            examples=[
                [
                    "Bass Metalcore Thrash Metal Furious bright vocal male Angry aggressive vocal Guitar",
                    """[verse]
Step back cause I'll ignite
Won't quit without a fight
No escape, gear up, it's a fierce fight
Brace up, raise your hands up and light
Fear the might. Step back cause I'll ignite
Won't back down without a fight
It keeps going and going, the heat is on.

[chorus]
Hot flame. Hot flame.
Still here, still holding aim
I don't care if I'm bright or dim: nah.
I've made it clear, I'll make it again
All I want is my crew and my gain.
I'm feeling wild, got a bit of rebel style.
Locked inside my mind, hot flame.
                    """
                ],
                [
                    "rap piano street tough piercing vocal hip-hop synthesizer clear vocal male",
                    """[verse]
Woke up in the morning, sun is shining bright
Chasing all my dreams, gotta get my mind right
City lights are fading, but my vision's clear
Got my team beside me, no room for fear
Walking through the streets, beats inside my head
Every step I take, closer to the bread
People passing by, they don't understand
Building up my future with my own two hands

[chorus]
This is my life, and I'm aiming for the top
Never gonna quit, no, I'm never gonna stop
Through the highs and lows, I'mma keep it real
Living out my dreams with this mic and a deal
                    """
                ]
            ],
            inputs=[genre_txt, lyrics_txt],
            outputs=[music_out, vocal_out, instrumental_out],
            cache_examples=True,
            cache_mode="eager",
            fn=infer
        )

    submit_btn.click(
        fn=infer,
        inputs=[genre_txt, lyrics_txt, num_segments, max_new_tokens],
        outputs=[music_out, vocal_out, instrumental_out]
    )
    gr.Markdown("## Call for Contributions\nIf you find this space interesting please feel free to contribute.")
demo.queue().launch(show_error=True)
