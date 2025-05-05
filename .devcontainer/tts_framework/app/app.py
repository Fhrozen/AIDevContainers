from copy import deepcopy
import json
import random
import os
import re

import librosa
import numpy as np
import torch

import streamlit as st
from streamlit_advanced_audio import audix

from app_code import SynthFactory


torch.classes.__path__ = [] # add this line to manually set it to empty.


AVAIL_TTS_MODELS = {
    0: {"label": "kokoro/v1.0.0-onnx", "deffunc": ""},
    1: {"label": "bark", "deffunc": ""}
}

AVAIL_PROMPT_MODELS = {
    0: {"label": "musicgen/small", "deffunc": ""},
}


# Specific for Gemini Citation
CITATION_PATTERN = r"\[\d+(?:,\s*\d+)*(?:-\d+)?\]"


def process_wav(model, text, kwargs):
    audio_array, duration, metrics = model.generate(
        text,
        **kwargs
    )
    samplerate = model.samplerate
    audio_array = librosa.resample(audio_array, orig_sr=samplerate, target_sr=16000)
    return audio_array


def generate_wav(routine):
    st.session_state[f"{routine}_use_counter"] = 0
    text = st.session_state.get(f"text_{routine}")
    if text is None or len(text) == 0:
        return
    model = st.session_state.get(f"{routine}_model_class", None)
    if model is None:
        return
    if routine == "tts":
        voice_names = st.session_state.tts_voice_value
        model.set_language(st.session_state.tts_lang_value)
        model.set_flavor(st.session_state.tts_flavor_value)
        kwargs = {
            "voice_names": [voice_names],
            "speed": st.session_state.tts_speed
        }
    elif routine == "prompt":
        kwargs = {
            "duration": st.session_state.prompt_dur,
            "guidance": st.session_state.prompt_guidance
        }
    audio_array = process_wav(model, text, kwargs)
    st.session_state[f"{routine}_array"] = audio_array
    st.session_state.samplerate = 16000


def tts_model():
    model_idx = st.session_state.get("model_idx", 0)
    model = SynthFactory.create_model(AVAIL_TTS_MODELS[model_idx]["label"])
    model.initialize()
    st.session_state.tts_model_class = model

    _flavors = model.list_flavors()
    _langs = model.list_languages()
    _voices = model.list_voices()
    st.session_state.tts_langs = _langs
    st.session_state.tts_voices = _voices
    st.session_state.tts_flavors = _flavors
    st.session_state.tts_lang_value = _langs[0]
    st.session_state.tts_voice_value = _voices[0]
    st.session_state.tts_flavor_value = _flavors[0]
    return


def prompt_model():
    prompt_idx = st.session_state.get("prompt_idx", 0)
    model = SynthFactory.create_model(AVAIL_PROMPT_MODELS[prompt_idx]["label"])
    model.initialize()
    st.session_state.prompt_model_class = model
    return


@st.fragment
def generate_player(key):
    samplerate = st.session_state.get("samplerate", 44100)
    array = st.session_state.get(f"{key}_array", np.zeros([16000]))
    audix(array, sample_rate=samplerate, key=f"{key}_player")


@st.fragment
def properties_tts():
    langs = st.session_state.get("tts_langs", [])
    voices = st.session_state.get("tts_voices", [])
    flavors = st.session_state.get("tts_flavors", [])
    with st.expander("See Configuration"):
        st.selectbox("Model Flavor", flavors, key="tts_flavor_value", index=0)
        st.selectbox("Voices", voices, key="tts_voice_value", index=0)
        st.selectbox("Language", langs, key="tts_lang_value", index=0)
        st.slider("Speed", 0.5, 2.0, 1.0, 0.1, key="tts_speed")


@st.fragment
def properties_prompt():
    with st.expander("See Configuration"):
        st.slider("Duration", 1, 30, 10, 1, key="prompt_dur")
        st.slider("Guidance", 1, 10, 3, 1, key="prompt_guidance")


def unload_model(routine):
    counter = st.session_state.get(f"{routine}_use_counter", 0)
    # print(counter)
    st.session_state[f"{routine}_use_counter"] = counter + 1
    if counter == 30:
        st.session_state[f"{routine}_model_class"].unload_model()
        st.session_state[f"{routine}_use_counter"] = 0


def tts_segment():
    st.markdown(
        """
        **Text-to-Speech**

        Generation with Kokoro-TTS (maybe later other TTS models)
        """
    )
    row1_1, _rspace, row1_2 = st.columns(
        (1, 0.02, 0.5)
    )
    # st.session_state.text_tts = "It was the best of times, it was the worst of times, it was the age of " + \
    #     "wisdom, it was the age of foolishness, it was the epoch of belief, it " + \
    #     "was the epoch of incredulity, it was the season of Light, it was the " + \
    #     "season of Darkness, it was the spring of hope, it was the winter of " + \
    #     "despair, (...)"
    model_idx = st.session_state.get("model_idx", None)
    if model_idx is None:
        tts_model()

    with row1_1:
        st.text_area("Enter Text", height=200, key="text_tts", label_visibility="collapsed")
        generate_player("tts")

    with row1_2:
        st.selectbox(
            "Model",
            range(len(AVAIL_TTS_MODELS)),
            format_func=lambda x: AVAIL_TTS_MODELS[x]["label"],
            key="model_idx",
            on_change=tts_model
        )
        properties_tts()
        st.button("Generate Speech", type="primary", on_click=generate_wav, args=("tts",))
        _tts_model = st.session_state.get("tts_model_class", None)
        run_every = None
        if _tts_model is not None:
            run_every = 1 if _tts_model._loaded else None

        @st.fragment(run_every = run_every)
        def unload_tts_model():
            unload_model("tts")

        unload_tts_model()
    return


def prompt_segment():
    st.markdown(
        """
        **Prompt-guided Music track generation**

        Generation with MusicGen
        """
    )
    row1_1, _rspace, row1_2 = st.columns(
        (1, 0.02, 0.5)
    )
    # st.session_state.text_tts = "It was the best of times, it was the worst of times, it was the age of " + \
    #     "wisdom, it was the age of foolishness, it was the epoch of belief, it " + \
    #     "was the epoch of incredulity, it was the season of Light, it was the " + \
    #     "season of Darkness, it was the spring of hope, it was the winter of " + \
    #     "despair, (...)"
    model_idx = st.session_state.get("prompt_model_idx", None)
    if model_idx is None:
        prompt_model()

    with row1_1:
        st.text_area("Enter Text", height=200, key="text_prompt", label_visibility="collapsed")
        generate_player("prompt")

    with row1_2:
        st.selectbox(
            "Model",
            range(len(AVAIL_PROMPT_MODELS)),
            format_func=lambda x: AVAIL_PROMPT_MODELS[x]["label"],
            key="prompt_model_idx",
            on_change=prompt_model
        )
        properties_prompt()
        st.button("Generate Track", key="prompt_gen", type="primary", on_click=generate_wav, args=("prompt",))
    return


def process_pod_text(is_return = False):
    st.session_state.pod_message = ""
    text = st.session_state.get(f"text_inp_pod")
    if text is None or len(text) == 0:
        return
    speakers = st.session_state.get("speakers_pod", None)
    if speakers is None:
        speakers = {}
    else:
        speakers = update_pod_speakers()
    lines = []
    text = text.replace("\n*", "*")
    voices = st.session_state.tts_voices
    langs = st.session_state.tts_langs
    for line in text.split("\n"):
        line = line.split("**", 2)
        if len(line) == 2:
            _, maybe_spk = line
            content = ""
        elif len(line) == 3:
            _, maybe_spk, content = line
        else:
            if "speaker" in line[0][:10]:
                maybe_spk, content = line[0].split(":", 1)
            else:
                continue
        if "music" in maybe_spk.lower():
            maybe_spk = "music"
            if "(" in content:
                content = ""
            if len(content) < 1:
                content = random.choice(musicgen_prompts)
            if maybe_spk not in speakers:
                speakers[maybe_spk] = {"duration": 5}
        elif maybe_spk.startswith("[") or ":" in maybe_spk:
            maybe_spk = maybe_spk.replace("[speaker]", "").replace(":", "").strip()
            maybe_spk = maybe_spk.strip("]").strip("[")
            if maybe_spk not in speakers:
                speakers[maybe_spk] = {
                    "voice_names": random.choice(voices),
                    "language": langs[0]
                }
            content = content.replace("**", "")
            content = content.replace("*", "")
            content = re.sub(CITATION_PATTERN, "", content)
        else:
            continue
        lines.append({"speaker": maybe_spk, "content": content})
    script = json.dumps({"lines": lines, "speakers": speakers}, indent=2)
    st.session_state.text_fmt_pod = script
    st.session_state.speakers_pod = speakers
    if is_return:
        return script


def update_pod_speakers():
    speakers = st.session_state.get("speakers_pod", {})
    for speaker in speakers:
        if speaker == "music":
            speakers[speaker] = {
                "duration": st.session_state.pod_dur,
                "guidance": st.session_state.pod_guidance,
            }
        else:
            speakers[speaker] = {
                "voice_names" : st.session_state.get(f"pod_voice_{speaker}"),
                "language" : st.session_state.get(f"pod_lang_{speaker}"),
                "speed" : st.session_state.get(f"pod_speed_{speaker}"),
            }
            
    st.session_state.speakers_pod = speakers
    return speakers


def generate_pod():
    # Generates the podcast audio using the script in dictionary format.
    script = st.session_state.get("text_fmt_pod", None)
    if script is None or len(script) < 1:
        script = process_pod_text(True)
    if script is None:
        return
    st.session_state.pod_message = ""
    try:
        script = json.loads(script)
    except:
        st.session_state.pod_message = "Error loading formated the script. Please check or analyze it once more."
        return
    
    speakers = update_pod_speakers()
    model_tts = st.session_state.get("tts_model_class", None)
    model_music = st.session_state.get("prompt_model_class", None)
    generated_wav = []
    total_lines = float(len(script["lines"]))
    progress_bar = st.session_state.pod_bar
    progress_bar.progress(0, text="Generating wav sequence")
    for idx, line in enumerate(script["lines"]):
        text = line["content"]
        spk = line["speaker"]
        kwargs = deepcopy(speakers[spk])
        if spk == "music":
            _model = model_music            
        else:
            _model = model_tts
            lang = kwargs.pop("language")
            _model.set_language(lang)

        wav = process_wav(_model, text, kwargs)
        if spk == "music":
            wav = wav * np.linspace(1.0, 0.0, wav.shape[0])
        else:
            wav = np.pad(wav, [0, int(16000 * 0.5)])
        generated_wav.append(wav)
        progress_bar.progress(int((idx + 1) * 100 / total_lines), text="Generating wav sequence")
    progress_bar.empty()
    st.session_state.samplerate = 16000
    st.session_state.podcast_array = np.concat(generated_wav, axis=0)
    script["speakers"] = speakers
    st.session_state.text_fmt_pod = json.dumps(script, indent=2)
    return


def properties_pod_speakers(tab, speaker, cfgs):
    if speaker == "music":
        tab.slider("Duration", 1, 30, cfgs.get("duration", 1), 1, key="pod_dur")
        tab.slider("Guidance", 1, 10, cfgs.get("guidance", 3), 1, key="pod_guidance")
    else:
        langs = st.session_state.tts_langs
        voices = st.session_state.tts_voices
        lng_idx = langs.index(cfgs["language"])
        voi_idx = voices.index(cfgs["voice_names"])
        tab.selectbox("Voices", voices, key=f"pod_voice_{speaker}", index=voi_idx)
        tab.selectbox("Language", langs, key=f"pod_lang_{speaker}", index=lng_idx)
        tab.slider("Speed", 0.5, 2.0, cfgs.get("speed", 1.0), 0.1, key=f"pod_speed_{speaker}")
    pass


def clear_pod_texts():
    st.session_state.text_inp_pod = ""
    st.session_state.text_fmt_pod = ""
    st.session_state.speakers_pod = {}


def podcast_segment():
    st.markdown(
        """
        **AI-based Podcast Synthesing**

        Generate audio podcast from script using TTS and MusicGen Models.
        """
    )
    row1, _rspace, row2 = st.columns((1, 0.02, 0.5))
    # st.session_state.text_tts = "It was the best of times, it was the worst of times, it was the age of " + \
    #     "wisdom, it was the age of foolishness, it was the epoch of belief, it " + \
    #     "was the epoch of incredulity, it was the season of Light, it was the " + \
    #     "season of Darkness, it was the spring of hope, it was the winter of " + \
    #     "despair, (...)"
    # model_idx = st.session_state.get("prompt_model_idx", None)
    # if model_idx is None:
    #     prompt_model()

    with row1:
        row1_1, _rspace, row1_2 = st.columns((1, 0.02, 1))
        row1_1.text_area("Raw script:", height=300, key="text_inp_pod")
        row1_1.button("Clear Text", key="pod_clear", type="primary", on_click=clear_pod_texts)

        row1_2.text_area("Formated script:", height=300, key="text_fmt_pod")
        row1_2.button("Analyze Text", key="pod_analyze", type="primary", on_click=process_pod_text)
        message = st.session_state.get("pod_message", "")
        row1_2.write(message)
        generate_player("podcast")

    with row2:
        # st.selectbox(
        #     "Model",
        #     range(len(AVAIL_PROMPT_MODELS)),
        #     format_func=lambda x: AVAIL_PROMPT_MODELS[x]["label"],
        #     key="prompt_model_idx",
        #     on_change=prompt_model
        # )
        # properties_prompt()
        st.button("Generate Audio", key="pod_gen", type="primary", on_click=generate_pod)
        pod_bar = st.progress(0, text="Generating Podcast audio")
        st.session_state.pod_bar = pod_bar
        speakers = st.session_state.get("speakers_pod", {})
        placeholder = st.empty()
        pod_bar.empty()
        if len(speakers) > 0:
            with placeholder.expander("Speakers configuration:"):
                tabs = st.tabs(list(speakers))
                for idx, (spk, cfgs) in enumerate(speakers.items()):
                    properties_pod_speakers(tabs[idx], spk, cfgs)
        else:
            placeholder.empty()
    return


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Sound Synthesizer")
    st.markdown(
        """ 
        """
    )

    prompt_file = os.path.join(os.path.dirname(__file__), "prompts_musicgen.txt")
    with open(prompt_file, "r", encoding="utf-8") as reader:
        musicgen_prompts = [x for x in reader.read().split("\n") if len(x) > 1]
    tab1, tab2, tab3 = st.tabs(["Text-to-Speech", "Prompt Synthesize", "Podcast"])
    with tab1:
        tts_segment()

    with tab2:
        prompt_segment()

    with tab3:
        podcast_segment()
