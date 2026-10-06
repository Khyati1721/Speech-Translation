import streamlit as st
import whisper
import tempfile
import os
import torch

from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================================================
# PAGE SETTINGS
# =========================================================

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
    "<h6 style='text-align: center; color: grey;'>"
    "By KC"
    "</h6>",
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
# NLLB-200
# Supports around 200 languages
# =========================================================

NLLB_MODEL = "facebook/nllb-200-distilled-600M"


# =========================================================
# LANGUAGES
# =========================================================

LANGUAGES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng",
    "Gujarati": "guj_Gujr",
    "Marathi": "mar_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Punjabi": "pan_Guru",
    "Urdu": "urd_Arab",
    "Assamese": "asm_Beng",
    "Nepali": "npi_Deva",
    "Sanskrit": "san_Deva",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn",
    "Russian": "rus_Cyrl",
    "Ukrainian": "ukr_Cyrl",
    "Polish": "pol_Latn",
    "Dutch": "nld_Latn",
    "Swedish": "swe_Latn",
    "Danish": "dan_Latn",
    "Norwegian": "nob_Latn",
    "Finnish": "fin_Latn",
    "Greek": "ell_Grek",
    "Czech": "ces_Latn",
    "Slovak": "slk_Latn",
    "Hungarian": "hun_Latn",
    "Romanian": "ron_Latn",
    "Bulgarian": "bul_Cyrl",
    "Croatian": "hrv_Latn",
    "Serbian": "srp_Cyrl",
    "Slovenian": "slv_Latn",
    "Bosnian": "bos_Latn",
    "Turkish": "tur_Latn",
    "Arabic": "arb_Arab",
    "Hebrew": "heb_Hebr",
    "Persian": "pes_Arab",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Chinese Simplified": "zho_Hans",
    "Chinese Traditional": "zho_Hant",
    "Vietnamese": "vie_Latn",
    "Thai": "tha_Thai",
    "Indonesian": "ind_Latn",
    "Malay": "zsm_Latn",
    "Filipino": "tgl_Latn",
    "Swahili": "swh_Latn",
    "Afrikaans": "afr_Latn",
    "Amharic": "amh_Ethi",
    "Welsh": "cym_Latn",
    "Irish": "gle_Latn",
    "Catalan": "cat_Latn",
    "Galician": "glg_Latn",
    "Basque": "eus_Latn",
    "Icelandic": "isl_Latn",
    "Estonian": "est_Latn",
    "Latvian": "lvs_Latn",
    "Lithuanian": "lit_Latn",
    "Albanian": "als_Latn",
    "Armenian": "hye_Armn",
    "Azerbaijani": "azj_Latn",
    "Georgian": "kat_Geor",
    "Kazakh": "kaz_Cyrl",
    "Uzbek": "uzn_Latn",
    "Tajik": "tgk_Cyrl",
    "Kyrgyz": "kir_Cyrl",
    "Turkmen": "tuk_Latn",
    "Mongolian": "khk_Cyrl",
    "Burmese": "mya_Mymr",
    "Khmer": "khm_Khmr",
    "Lao": "lao_Laoo",
    "Sinhala": "sin_Sinh",
    "Pashto": "pbt_Arab",
    "Somali": "som_Latn",
    "Yoruba": "yor_Latn",
    "Zulu": "zul_Latn",
    "Xhosa": "xho_Latn",
    "Haitian Creole": "hat_Latn",
    "Esperanto": "epo_Latn"
}


# =========================================================
# LOAD WHISPER
# =========================================================

@st.cache_resource
def load_whisper():

    return whisper.load_model("base")


model = load_whisper()


# =========================================================
# LOAD NLLB
# =========================================================

@st.cache_resource
def load_translation_model():

    tokenizer = AutoTokenizer.from_pretrained(
        NLLB_MODEL
    )

    translation_model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL
    )

    translation_model.eval()

    return tokenizer, translation_model


# =========================================================
# TRANSLATE
# =========================================================

def translate_text(
    text,
    source_language,
    target_language
):

    tokenizer, translation_model = load_translation_model()

    tokenizer.src_lang = source_language

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        translated_tokens = translation_model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                target_language
            ),
            max_length=512
        )

    result = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )

    return result[0]


# =========================================================
# TEXT TO SPEECH
# =========================================================

def text_to_speech(
    text,
    language
):

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
# TTS LANGUAGE CODES
# =========================================================

TTS_CODES = {
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
    "Ukrainian": "uk",
    "Polish": "pl",
    "Dutch": "nl",
    "Swedish": "sv",
    "Danish": "da",
    "Norwegian": "no",
    "Finnish": "fi",
    "Greek": "el",
    "Czech": "cs",
    "Slovak": "sk",
    "Hungarian": "hu",
    "Romanian": "ro",
    "Bulgarian": "bg",
    "Croatian": "hr",
    "Serbian": "sr",
    "Slovenian": "sl",
    "Turkish": "tr",
    "Arabic": "ar",
    "Hebrew": "he",
    "Persian": "fa",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese Simplified": "zh-CN",
    "Chinese Traditional": "zh-TW",
    "Vietnamese": "vi",
    "Thai": "th",
    "Indonesian": "id",
    "Malay": "ms",
    "Filipino": "tl",
    "Swahili": "sw",
    "Afrikaans": "af",
    "Amharic": "am",
    "Welsh": "cy",
    "Irish": "ga",
    "Catalan": "ca",
    "Galician": "gl",
    "Basque": "eu",
    "Icelandic": "is",
    "Estonian": "et",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Albanian": "sq",
    "Armenian": "hy",
    "Azerbaijani": "az",
    "Georgian": "ka",
    "Kazakh": "kk",
    "Uzbek": "uz",
    "Tajik": "tg",
    "Kyrgyz": "ky",
    "Turkmen": "tk",
    "Mongolian": "mn",
    "Burmese": "my",
    "Khmer": "km",
    "Lao": "lo",
    "Sinhala": "si",
    "Pashto": "ps",
    "Somali": "so",
    "Yoruba": "yo",
    "Zulu": "zu",
    "Xhosa": "xh",
    "Haitian Creole": "ht",
    "Esperanto": "eo"
}


# =========================================================
# AUDIO TRANSCRIPTION
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
            "🔄 Transcribing audio..."
        )

        result = model.transcribe(
            file_path
        )

        st.success(
            "✅ Transcription Complete"
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
# INPUT
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
# TEXT
# =========================================================

if inp == "Text":

    data = c1.text_area(
        "Enter Text Here",
        height=180
    )


# =========================================================
# MICROPHONE
# =========================================================

elif inp == "MIC":

    recorded_file = c1.audio_input(
        "🎤 Record Audio"
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
                data,
                height=180
            )


# =========================================================
# AUDIO FILE
# =========================================================

else:

    uploaded_file = c1.file_uploader(
        "📁 Upload Audio File",
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
                data,
                height=180
            )


# =========================================================
# SOURCE LANGUAGE
# =========================================================

source_language_name = c1.selectbox(
    "Source Language",
    list(LANGUAGES.keys())
)

source_language = LANGUAGES[
    source_language_name
]


# =========================================================
# TARGET LANGUAGE
# =========================================================

target_language_name = c1.selectbox(
    "Output Language",
    list(LANGUAGES.keys())
)

target_language = LANGUAGES[
    target_language_name
]


# =========================================================
# TRANSLATE BUTTON
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

    elif source_language == target_language:

        st.session_state[
            "translated_text"
        ] = source_text

        st.session_state[
            "translated_language"
        ] = target_language_name

        c2.text_area(
            "Translated Text",
            source_text,
            height=180
        )

        st.success(
            "✅ Source and output languages are the same."
        )

    else:

        try:

            with st.spinner(
                "🌐 Translating locally..."
            ):

                translated_text = translate_text(
                    source_text,
                    source_language,
                    target_language
                )

            st.session_state[
                "translated_text"
            ] = translated_text

            st.session_state[
                "translated_language"
            ] = target_language_name

            c2.text_area(
                "Translated Text",
                translated_text,
                height=180
            )

            st.success(
                "✅ Translation Complete"
            )

        except Exception as e:

            st.error(
                f"❌ Translation Error: {e}"
            )


# =========================================================
# TEXT TO SPEECH
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

    elif saved_language != target_language_name:

        st.warning(
            "⚠️ Please translate again after changing the output language."
        )

    elif target_language_name not in TTS_CODES:

        st.warning(
            "⚠️ Speech is not available for this language through gTTS."
        )

    else:

        try:

            with st.spinner(
                "🔊 Creating speech..."
            ):

                audio_file = text_to_speech(
                    translated_text,
                    TTS_CODES[target_language_name]
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
