import numpy as np
import torch

import streamlit as st
from streamlit_advanced_audio import audix

from tts_factory import TTSFactory


torch.classes.__path__ = [] # add this line to manually set it to empty.


AVAIL_MODELS = {
    0: {"label": "kokoro/v1.0.0-onnx", "deffunc": ""},
    1: {"label": "bark", "deffunc": ""}
}


def generate_wav():
    text = st.session_state.text_tts
    if text is None or len(text) == 0:
        return
    model = st.session_state.get("tts_model_class", None)
    if model is None:
        return
    voice_names = model.list_voices()[0]
    speed = 1.0
    audio_array, duration, metrics = model.generate_speech(
        text,
        [voice_names],
        speed
    )
    st.session_state.array = audio_array
    st.session_state.samplerate = 24000
    # st.write(value)


def tts_model():
    model_idx = st.session_state.model_idx
    model = TTSFactory.create_model(AVAIL_MODELS[model_idx]["label"])
    model.initialize()
    voices = model.list_voices()
    langs = model.list_languages()
    flavors = model.list_flavors()
    st.session_state.tts_model_class = model
    return


@st.fragment
def generate_player():
    samplerate = st.session_state.get("samplerate", 44100)
    array = st.session_state.get("array", np.zeros([16000]))
    print(samplerate)
    audix(array, sample_rate=int(samplerate * 2.76))


def properties_tts():
    with st.empty():
        with st.expander("See explanation"):
            st.write('''
                The chart above shows some numbers I picked for you.
                I rolled actual dice for these, so they're *guaranteed* to
                be random.
            ''')
            st.image("https://static.streamlit.io/examples/dice.jpg")


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

    with row1_1:
        st.text_area("Enter Text", height=200, key="text_tts", label_visibility="hidden")
        generate_player()

    with row1_2:
        st.selectbox(
            "Model",
            range(len(AVAIL_MODELS)),
            format_func=lambda x: AVAIL_MODELS[x]["label"],
            key="model_idx",
            on_change=tts_model,
            index=None
        )
        properties_tts()
        st.button("Generate Speech", type="primary", on_click=generate_wav)

    return


def prompt_segment():
    st.markdown(
        """
        **Prompt-based Sound Synthesize**

        Generation with Dia / Bark
        """
    )
    txt1 = st.text_area(
            "Enter prompt",
            "It was the best of times, it was the worst of times, it was the age of "
            "wisdom, it was the age of foolishness, it was the epoch of belief, it "
            "was the epoch of incredulity, it was the season of Light, it was the "
            "season of Darkness, it was the spring of hope, it was the winter of "
            "despair, (...)",
            height=200
        )
    return


def podcast_segment():
    st.markdown(
        """
        **Podcast Synthesize**
        """
    )
    txt2 = st.text_area(
        "Enter podcast text",
        "It was the best of times, it was the worst of times, it was the age of "
        "wisdom, it was the age of foolishness, it was the epoch of belief, it "
        "was the epoch of incredulity, it was the season of Light, it was the "
        "season of Darkness, it was the spring of hope, it was the winter of "
        "despair, (...)",
        height=200
    )
    return


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Sound Synthesizer")
    st.markdown(
        """ 
        """
    )

    tab1, tab2, tab3 = st.tabs(["Text-to-Speech", "Prompt Synthesize", "Podcast"])
    with tab1:
        tts_segment()

    with tab2:
        prompt_segment()

    with tab3:
        podcast_segment()
