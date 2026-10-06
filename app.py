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
    "<h1 style='text-align: center; color: grey;'>Speech to Speech Translation</h1>",
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
    .stSelectbox div {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# TEXT TO SPEECH
# --------------------------------------------------

def text_to_speech(text, accent):
    output_file = "output.mp3"

    tts = gTTS(text=text, lang=accent)
    tts.save(output_file)

    return output_file


# --------------------------------------------------
# WHISPER TRANSCRIPTION
# --------------------------------------------------

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

        return result["text"]

    except Exception as e:

        st.error(f"❌ Error: {e}")
        return ""

    finally:

        if os.path.exists(file_path):
            os.remove(file_path)


# --------------------------------------------------
# LOAD WHISPER MODEL
# --------------------------------------------------

@st.cache_resource
def load_whisper_model():

    return whisper.load_model("base")


model = load_whisper_model()


# --------------------------------------------------
# GET LANGUAGES
# --------------------------------------------------

@st.cache_data(ttl=86400)
def get_languages():

    return gt().get_supported_languages(as_dict=True)


try:

    languages = get_languages()

except Exception:

    st.error(
        "⚠️ Could not load translation languages. "
        "Google Translate may be temporarily rate-limiting requests."
    )

    st.stop()


language_names = list(languages.keys())


# --------------------------------------------------
# TRANSLATION
# --------------------------------------------------

@st.cache_data(ttl=3600)
def translate_text(text, target):

    translator = gt(target=target)

    return translator.translate(text)


# --------------------------------------------------
# UI
# --------------------------------------------------

c1, c3, c2 = st.columns(3)

inp = c1.selectbox(
    "Choose Input Format",
    ("Text", "MIC", "Audio File")
)

data = ""
recorded_file = None
uploaded_file = None


# --------------------------------------------------
# INPUT
# --------------------------------------------------

if inp == "Text":

    data = c1.text_area("Enter Text Here")


elif inp == "MIC":

    recorded_file = c1.audio_input("Record Audio")

    if recorded_file and c2.button("Transcribe"):

        data = transcribe(recorded_file)

        c1.text_area(
            "Transcribed Text",
            data
        )


else:

    uploaded_file = c1.file_uploader(
        "Upload Audio File"
    )

    if uploaded_file and c2.button("🎤 Transcribe"):

        data = transcribe(uploaded_file)

        c1.text_area(
            "Transcribed Text",
            data
        )


# --------------------------------------------------
# OUTPUT LANGUAGE
# --------------------------------------------------

option = c1.selectbox(
    "Output Language",
    language_names
)


# --------------------------------------------------
# TRANSLATE BUTTON
# --------------------------------------------------

if c2.button("🌐 Translate Text"):

    # MIC
    if inp == "MIC" and recorded_file:

        data = transcribe(recorded_file)

        c1.text_area(
            "Transcribed Text",
            data
        )

    # AUDIO FILE
    elif inp == "Audio File" and uploaded_file:

        data = transcribe(uploaded_file)

        c1.text_area(
            "Transcribed Text",
            data
        )

    # Check empty text
    if not data.strip():

        st.warning("Please enter or provide some text.")

    else:

        try:

            translated_text = translate_text(
                data,
                option
            )

            c2.text_area(
                "Translated Text",
                translated_text
            )

            # Save translation so we don't translate again
            st.session_state["translated_text"] = translated_text

            st.success("✅ Successfully Translated")

        except Exception as e:

            st.error(
                "❌ Translation failed. "
                "Google Translate may be temporarily rate-limiting requests."
            )


# --------------------------------------------------
# TEXT TO SPEECH
# --------------------------------------------------

if c2.button("🔊 Convert To Speech"):

    # Use existing translation if available
    translated_text = st.session_state.get(
        "translated_text",
        ""
    )

    # If translation doesn't exist, translate first
    if not translated_text:

        if inp == "MIC" and recorded_file:

            data = transcribe(recorded_file)

        elif inp == "Audio File" and uploaded_file:

            data = transcribe(uploaded_file)

        if not data.strip():

            st.warning("Please enter or provide some text.")

        else:

            try:

                translated_text = translate_text(
                    data,
                    option
                )

                st.session_state[
                    "translated_text"
                ] = translated_text

            except Exception:

                st.error(
                    "❌ Translation failed. "
                    "Please try again later."
                )

    # Convert translated text to speech
    if translated_text.strip():

        try:

            audio_file = text_to_speech(
                translated_text,
                languages[option]
            )

            c2.audio(
                audio_file,
                format="audio/mp3"
            )

        except Exception as e:

            st.error(
                f"❌ Text-to-speech error: {e}"
            )
