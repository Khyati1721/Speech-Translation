import streamlit as st
from deep_translator import GoogleTranslator as gt
from gtts import gTTS
import whisper
import tempfile
import os

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


# Languages
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Urdu": "ur",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh-CN"
}


# Load Whisper only once
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")


model = load_whisper()


# Translate text
@st.cache_data(ttl=3600)
def translate_text(text, target):
    translator = gt(
        source="auto",
        target=target
    )
    return translator.translate(text)


# Text to speech
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


# Transcribe audio
def transcribe(audio_file):

    extension = audio_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{extension}"
    ) as tmpfile:

        file_path = tmpfile.name
        tmpfile.write(audio_file.read())

    try:

        st.write("🔄 Transcribing... Please wait!")

        result = model.transcribe(file_path)

        st.success("✅ Transcription Complete!")

        return result["text"].strip()

    except Exception as e:

        st.error(f"❌ Transcription Error: {e}")

        return ""

    finally:

        if os.path.exists(file_path):
            os.remove(file_path)


# Create columns
c1, c3, c2 = st.columns(3)


# Input format
inp = c1.selectbox(
    "Choose Input Format",
    ("Text", "MIC", "Audio File")
)

data = ""


# Text input
if inp == "Text":

    data = c1.text_area(
        "Enter Text Here"
    )


# Microphone input
elif inp == "MIC":

    recorded_file = c1.audio_input(
        "Record Audio"
    )

    if recorded_file:

        if c2.button("🎤 Transcribe"):

            data = transcribe(
                recorded_file
            )

            st.session_state["source_text"] = data

            c1.text_area(
                "Transcribed Text",
                data
            )


# Audio file input
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

        if c2.button("🎤 Transcribe"):

            data = transcribe(
                uploaded_file
            )

            st.session_state["source_text"] = data

            c1.text_area(
                "Transcribed Text",
                data
            )


# Output language
option = c1.selectbox(
    "Output Language",
    list(LANGUAGES.keys())
)

target_language = LANGUAGES[option]


# Translate button
if c2.button("🌐 Translate Text"):

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

            translated_text = translate_text(
                source_text,
                target_language
            )

            st.session_state["translated_text"] = translated_text

            st.session_state["translated_language"] = option

            c2.text_area(
                "Translated Text",
                translated_text
            )

            st.success(
                "✅ Successfully Translated"
            )

        except Exception:

            st.error(
                "❌ Google Translate is currently rate-limiting this app."
            )

            st.info(
                "Please wait a little while and try again."
            )


# Convert translated text to speech
if c2.button("🔊 Convert To Speech"):

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

    elif saved_language != option:

        st.warning(
            "⚠️ You changed the output language. "
            "Please translate again."
        )

    else:

        try:

            audio_file = text_to_speech(
                translated_text,
                target_language
            )

            c2.audio(
                audio_file,
                format="audio/mp3"
            )

        except Exception as e:

            st.error(
                f"❌ Text-to-speech error: {e}"
            )
