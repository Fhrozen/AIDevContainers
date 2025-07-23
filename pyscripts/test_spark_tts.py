import json
import numpy as np
import re
import argparse
import os

import onnxruntime as ort
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, PretrainedConfig, GenerationConfig

import soundfile as sf


GENDER_MAP = {
    "female": 0,
    "male": 1,
}

LEVELS_MAP = {
    "very_low": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "very_high": 4,
}

TASK_TOKEN_MAP = {
    "vc": "<|task_vc|>",
    "tts": "<|task_tts|>",
    "asr": "<|task_asr|>",
    "s2s": "<|task_s2s|>",
    "t2s": "<|task_t2s|>",
    "understand": "<|task_understand|>",
    "caption": "<|task_cap|>",
    "controllable_tts": "<|task_controllable_tts|>",
    "prompt_tts": "<|task_prompt_tts|>",
    "speech_edit": "<|task_edit|>",
}


def process_prompt(
    text: str,
    prompt_speech_path,
    audio_tokenizer,
    prompt_text: str = None,
):
    global_token_ids, semantic_token_ids = audio_tokenizer.tokenize(
        prompt_speech_path
    )
    global_tokens = "".join(
        [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
    )

    # Prepare the input tokens for the model
    if prompt_text is not None:
        semantic_tokens = "".join(
            [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
        )
        inputs = [
            TASK_TOKEN_MAP["tts"],
            "<|start_content|>",
            prompt_text,
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
            "<|start_semantic_token|>",
            semantic_tokens,
        ]
    else:
        inputs = [
            TASK_TOKEN_MAP["tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
        ]

    inputs = "".join(inputs)

    return inputs, global_token_ids


def process_prompt_control(gender, pitch, speed, text):
    gender_id = GENDER_MAP[gender]
    pitch_level_id = LEVELS_MAP[pitch]
    speed_level_id = LEVELS_MAP[speed]

    pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
    speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
    gender_tokens = f"<|gender_{gender_id}|>"

    attribte_tokens = "".join(
        [gender_tokens, pitch_label_tokens, speed_label_tokens]
    )

    control_tts_inputs = [
        TASK_TOKEN_MAP["controllable_tts"],
        "<|start_content|>",
        text,
        "<|end_content|>",
        "<|start_style_label|>",
        attribte_tokens,
        "<|end_style_label|>",
    ]

    return "".join(control_tts_inputs)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Spark TTS inference script")
    parser.add_argument("--text", type=str, required=True, help="Text for TTS generation")
    parser.add_argument("--prompt", type=str, help="Transcript of prompt audio")
    parser.add_argument("--gender_voice", type=str, default="male", help="Voice gender")
    parser.add_argument("--clone_voice", type=str, default=None, help="Path to voice clone file")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--flavor", type=str, default="q4", help="Model flavor: FP32, FP6, or quantized.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--pitch", type=str, default="moderate", help="Voice pitch.")
    parser.add_argument("--speed", type=str, default="moderate", help="Voice pitch.")
    return parser.parse_args()


def main():
    args = parse_arguments()

    use_gpu = False  # "CUDAExecutionProvider" in ort.get_available_providers()
    providers = [
        ("CUDAExecutionProvider", {"device_id": args.num_gpus - 1})
    ] if use_gpu else []
    providers.append("CPUExecutionProvider")
    
    work_dir = os.path.join(args.model_dir, "LLM")
    if os.path.exists(work_dir):
        config = PretrainedConfig.from_pretrained(work_dir)
        gen_config = GenerationConfig.from_pretrained(work_dir)

        suffix = "" if (
            args.flavor is None or args.flavor == ""
        ) else f"_{args.flavor}"
        model_path = os.path.join(work_dir, "onnx", f"model{suffix}.onnx")
        
        sess_options = ort.SessionOptions()
        ort_model = ort.InferenceSession(model_path, sess_options, providers=providers)
        llm_model = ORTModelForCausalLM(
            session=ort_model,
            config=config,
            generation_config=gen_config,
            use_io_binding=True,
            use_cache=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(work_dir)

    else:
        raise ValueError(f"{model_path} does not exist.")
    
    sess_options = ort.SessionOptions()
    audio_detokenizer = ort.InferenceSession(
        os.path.join(args.model_dir, "bicodec.onnx"),
        sess_options,
        providers=providers
    )

    # Process the prompt and clone_voice
    if args.clone_voice:
        raise NotImplementedError()
        print(f"Using voice clone: {args.clone_voice}")
        prompt, global_tokens = process_prompt(args.text, args.clone_voice, args.prompt)
    else:
        print(f"Using {args.gender_voice} voice ")
        prompt = process_prompt_control(args.gender_voice, args.pitch, args.speed, args.text)

    print(f"Using prompt: {prompt}")
    inputs = tokenizer([prompt], return_tensors="pt")

    generated_ids = llm_model.generate(
        **inputs,
        max_new_tokens=3000,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
    )
    # Trim the output tokens to remove the input tokens
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the generated tokens into text
    predicts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    semantic_tokens = np.array(
        [int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)]
    )[None]

    if args.clone_voice is None:
        global_token = np.array(
            [int(token) for token in re.findall(r"bicodec_global_(\d+)", predicts)]
        )[None, None]
    print(semantic_tokens.shape)
        
    wav = audio_detokenizer.run(
        ["audio"],
        {"semantic_tokens": semantic_tokens, "global_tokens": global_token}
    )
    print(wav.shape)
    return


# run: python test_spark_tts.py --text "Your text to synthesize" --model_dir "/path/to/model"
if __name__ == "__main__":
    main()
