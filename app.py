import streamlit as st
from gtts import gTTS
import whisper
import tempfile
import os
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


st.set_page_config(
    page_title="Simply! Translate",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    "<h1 style='text-align: center; color: grey;'>"
    "Speech to Speech Translation"
    "</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h6 style='text-align: center; color: grey;'>By KC</h6>",
    unsafe_allow_html=True
)

st.markdown("""
<style>
.stButton button {
    background-color: #4CAF50;
    color: white;
    font-size: 16px;
    padding: 10px;
    border-radius: 10px;
}

.stTextArea textarea {
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD WHISPER
# =========================================================

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")


model = load_whisper()


# =========================================================
# TRANSLATION MODELS
# =========================================================

TRANSLATION_MODELS = {
    "English → Hindi": {
        "model": "Helsinki-NLP/opus-mt-en-hi",
        "source": "en",
        "target": "hi"
    },

    "Hindi → English": {
        "model": "Helsinki-NLP/opus-mt-hi-en",
        "source": "hi",
        "target": "en"
    }
}


@st.cache_resource
def load_translation_model(model_name):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    translation_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    return tokenizer, translation_model


def translate_text(text, model_name):

    tokenizer, translation_model = load_translation_model(
        model_name
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        translated = translation_model.generate(
            **inputs,
            max_length=512
        )

    result = tokenizer.decode(
        translated[0],
        skip_special_tokens=True
    )

    return result


# =========================================================
# TEXT TO SPEECH
# =========================================================

def text_to_speech(text, language):

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    ).name

    tts = gTTS(
        text=text,
        lang=language
    )

    tts.save(output_file)

    return output_file


# =========================================================
# TRANSCRIBE AUDIO
# =========================================================

def transcribe(audio_file):

    extension = audio_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{extension}"
    ) as tmpfile:

        file_path = tmpfile.name
        tmpfile.write(audio_file.read())

    try:

        st.info(
            "🔄 Transcribing... Please wait!"
        )

        result = model.transcribe(
            file_path
        )

        st.success(
            "✅ Transcription Complete!"
        )

        return result["text"].strip()

    except Exception as e:

        st.error(
            f"❌ Transcription Error: {e}"
        )

        return ""

    finally:

        if os.path.exists(file_path):
            os.remove(file_path)


# =========================================================
# COLUMNS
# =========================================================

c1, c3, c2 = st.columns(3)


# =========================================================
# INPUT FORMAT
# =========================================================

inp = c1.selectbox(
    "Choose Input Format",
    (
        "Text",
        "MIC",
        "Audio File"
    )
)

data = ""


# =========================================================
# TEXT INPUT
# =========================================================

if inp == "Text":

    data = c1.text_area(
        "Enter Text Here"
    )


# =========================================================
# MICROPHONE INPUT
# =========================================================

elif inp == "MIC":

    recorded_file = c1.audio_input(
        "Record Audio"
    )

    if recorded_file:

        if c2.button(
            "🎤 Transcribe"
        ):

            data = transcribe(
                recorded_file
            )

            st.session_state[
                "source_text"
            ] = data

            c1.text_area(
                "Transcribed Text",
                data
            )


# =========================================================
# AUDIO FILE INPUT
# =========================================================

else:

    uploaded_file = c1.file_uploader(
        "Upload Audio File",
        type=[
            "wav",
            "mp3",
            "m4a",
            "ogg",
            "webm"
        ]
    )

    if uploaded_file:

        if c2.button(
            "🎤 Transcribe"
        ):

            data = transcribe(
                uploaded_file
            )

            st.session_state[
                "source_text"
            ] = data

            c1.text_area(
                "Transcribed Text",
                data
            )


# =========================================================
# LANGUAGE
# =========================================================

translation_option = c1.selectbox(
    "Translation Direction",
    list(TRANSLATION_MODELS.keys())
)

translation_settings = TRANSLATION_MODELS[
    translation_option
]

model_name = translation_settings["model"]

target_language = translation_settings["target"]


# =========================================================
# TRANSLATE
# =========================================================

if c2.button(
    "🌐 Translate Text"
):

    if inp == "Text":

        source_text = data.strip()

    else:

        source_text = st.session_state.get(
            "source_text",
            ""
        ).strip()

    if not source_text:

        st.warning(
            "⚠️ Please enter text or transcribe audio first."
        )

    else:

        try:

            with st.spinner(
                "🌐 Translating locally..."
            ):

                translated_text = translate_text(
                    source_text,
                    model_name
                )

            st.session_state[
                "translated_text"
            ] = translated_text

            st.session_state[
                "translated_language"
            ] = target_language

            c2.text_area(
                "Translated Text",
                translated_text
            )

            st.success(
                "✅ Translation Complete"
            )

        except Exception as e:

            st.error(
                f"❌ Translation Error: {e}"
            )


# =========================================================
# CONVERT TO SPEECH
# =========================================================

if c2.button(
    "🔊 Convert To Speech"
):

    translated_text = st.session_state.get(
        "translated_text",
        ""
    )

    saved_language = st.session_state.get(
        "translated_language",
        ""
    )

    if not translated_text:

        st.warning(
            "⚠️ Please translate the text first."
        )

    elif saved_language != target_language:

        st.warning(
            "⚠️ Please translate again before converting to speech."
        )

    else:

        try:

            with st.spinner(
                "🔊 Creating speech..."
            ):

                audio_file = text_to_speech(
                    translated_text,
                    target_language
                )

            c2.audio(
                audio_file,
                format="audio/mp3"
            )

            st.success(
                "✅ Speech Created"
            )

        except Exception as e:

            st.error(
                f"❌ Text-to-speech error: {e}"
            )
