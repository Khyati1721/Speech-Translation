import streamlit as st
from deep_translator import GoogleTranslator as gt
from gtts import gTTS
import whisper
import tempfile
import os



st.set_page_config(page_title='Simply! Translate',
                   page_icon='🌍',
                   layout='wide',
                   initial_sidebar_state='expanded')

st.markdown("<h1 style='text-align: center; color: grey;'>Speach to Speach Translation</h1>",
            unsafe_allow_html=True)

st.markdown("<h6 style='text-align: center; color: grey;'>By KC </h6>",
            unsafe_allow_html=True)

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


# Convert text to speech and save as MP3
def text_to_speech(text, accent):
    tts = gTTS(text=text, lang=accent)
    tts.save("output.mp3")
    return "output.mp3"

# Transcribe audio using Whisper
def transcribe(audio_file):
    extension = audio_file.name.split(".")[-1]

    # Store uploaded audio in a temporary file for Whisper
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmpfile:
        file_path = tmpfile.name
        tmpfile.write(audio_file.read())

    try:
        c1.write("🔄 Transcribing... Please wait!")
        result = model.transcribe(file_path)
        c1.success("✅ Transcription Complete!")
        return result['text']

    except Exception as e:
        st.error(f"❌ Error: {e}")
        return ""

    finally:
        # Remove temporary file after processing
        if os.path.exists(file_path):
            os.remove(file_path)



# Load Whisper model
model = whisper.load_model('base')

c1, c3, c2 = st.columns(3)
inp = c1.selectbox('Choose Input Format',('Text', 'MIC', 'Audio File'))

data = ""
recorded_file = None
uploaded_file = None



if inp == 'Text':
    data = c1.text_area('Enter Text Here')

elif inp == 'MIC':
    recorded_file = c1.audio_input('Record Audio')

    if recorded_file and c2.button('Transcribe'):
        data = transcribe(recorded_file)
        c1.text_area('Transcribed Text', data)

else:
    uploaded_file = c1.file_uploader('Upload Audio File')

    if uploaded_file and c2.button('🎤 Transcribe'):
        data = transcribe(uploaded_file)
        c1.text_area('Transcribed Text', data)

# Get all supported languages from Google Translator
languages = gt().get_supported_languages(as_dict=True)
language_names = list(languages.keys())

option = c1.selectbox("Output Language",language_names)



if c2.button('🌐 Translate Text'):

    if inp == 'MIC' and recorded_file:
        data = transcribe(recorded_file)
        c1.text_area('Transcribed Text', data)

    elif inp == 'Audio File' and uploaded_file:
        data = transcribe(uploaded_file)
        c1.text_area('Transcribed Text', data)

    if data == "":
        st.warning("Please **Enter Text**")

    else:
        translated_text = gt(target=option).translate(data)

        c2.text_area('Translated Text', translated_text)
        c2.success('Successfully Translated')



if c2.button('🔊 Convert To Speech'):

    if inp == 'MIC' and recorded_file:
        data = transcribe(recorded_file)
        c1.text_area('Transcribed Text', data)

    elif inp == 'Audio File' and uploaded_file:
        data = transcribe(uploaded_file)
        c1.text_area('Transcribed Text', data)

    if data:
        translated_text = gt(target=option).translate(data)
        if translated_text.strip():

            audio_file = text_to_speech(
                translated_text,
                languages[option]
            )
            c2.audio(audio_file, format='audio/mp3')