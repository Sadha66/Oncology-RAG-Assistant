
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from faster_whisper import WhisperModel
from gtts import gTTS
from io import BytesIO

import tempfile
import os


# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="Oncology RAG Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Oncology RAG Assistant")

st.caption(
    "Multilingual Oncology Assistant | "
    "FAISS + Ollama + Llama3 + Voice"
)


# ---------------------------
# LANGUAGE SETTINGS
# ---------------------------

LANGUAGES = {
    "English": {
        "code": "en",
        "name": "English"
    },
    "Tamil": {
        "code": "ta",
        "name": "Tamil"
    },
    "Hindi": {
        "code": "hi",
        "name": "Hindi"
    },
    "Telugu": {
        "code": "te",
        "name": "Telugu"
    },
    "Malayalam": {
        "code": "ml",
        "name": "Malayalam"
    },
    "Kannada": {
        "code": "kn",
        "name": "Kannada"
    }
}


# ---------------------------
# LOAD RESOURCES
# ---------------------------

@st.cache_resource
def load_resources():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "vectordb",
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = OllamaLLM(
        model="llama3",
        temperature=0,
        num_predict=400,
        repeat_penalty=1.15
    )

    return db, llm


# ---------------------------
# LOAD WHISPER
# ---------------------------

@st.cache_resource
def load_whisper():

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

    return model


# ---------------------------
# TRANSLATE QUESTION
# ---------------------------

def translate_question(llm, question):

    translation_prompt = f"""
Translate the following oncology question into English.

Rules:
- Preserve the exact medical meaning.
- Do not answer the question.
- Do not add any information.
- Return only the translated question.

Question:
{question}

English translation:
"""

    translated = llm.invoke(translation_prompt)

    return translated.strip()


# ---------------------------
# GENERATE MULTILINGUAL ANSWER
# ---------------------------

def generate_answer(llm, context, question, language):

    language_name = LANGUAGES[language]["name"]

    prompt = f"""
You are a careful oncology educational assistant.

Answer the user's exact question using ONLY the
provided medical document context.

IMPORTANT RULES:

1. Answer in {language_name}.
2. Preserve important medical terminology.
3. You may show important medical terms in English
   alongside their translation.
4. Do not change the medical meaning.
5. Do not invent medical facts.
6. Do not provide unsupported treatment advice.
7. Do not repeat sentences or phrases.
8. Answer only the exact question.
9. If the context does not contain the answer,
   say that the information is not available
   in the retrieved documents.
10. Use simple language.
11. Use bullet points where appropriate.
12. Keep the answer clear and relevant.

Retrieved Medical Context:

{context}

User's Question:

{question}

Answer in {language_name}:
"""

    return llm.invoke(prompt)


# ---------------------------
# SESSION STATE
# ---------------------------

defaults = {
    "loaded": False,
    "answer": None,
    "docs": [],
    "answer_audio": None,
    "current_question": "",
    "selected_language": "English"
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------
# LOAD RAG SYSTEM
# ---------------------------

if not st.session_state.loaded:

    if st.button("🚀 Load RAG System"):

        with st.spinner("Loading Models and Database..."):

            try:

                db, llm = load_resources()

                st.session_state.db = db
                st.session_state.llm = llm
                st.session_state.loaded = True

                st.success("System Loaded Successfully!")

                st.rerun()

            except Exception as e:

                st.error(f"Loading error: {e}")

    st.stop()


# ---------------------------
# LANGUAGE SELECTION
# ---------------------------

st.subheader("🌐 Select Your Language")

selected_language = st.selectbox(
    "Choose answer language",
    options=list(LANGUAGES.keys()),
    index=0,
    key="language_selector"
)


# ---------------------------
# QUESTION INPUT
# ---------------------------

st.subheader("Ask a Question")

col1, col2 = st.columns([8, 2])

with col1:

    question = st.text_input(
        "Type your question",
        placeholder="Ask your oncology question...",
        label_visibility="collapsed"
    )

with col2:

    audio_input = st.audio_input(
        "🎤 Speak",
        sample_rate=16000
    )


st.caption(
    "You can type or speak in English, Tamil, Hindi, "
    "Telugu, Malayalam or Kannada."
)


# ---------------------------
# ASK AI BUTTON
# ---------------------------

if st.button("🔍 Ask AI"):

    final_question = question.strip()

    # Clear previous results

    st.session_state.answer = None
    st.session_state.docs = []
    st.session_state.answer_audio = None
    st.session_state.current_question = ""

    st.session_state.selected_language = selected_language

    # -----------------------
    # VOICE RECOGNITION
    # -----------------------

    if not final_question and audio_input is not None:

        with st.spinner("🎤 Recognizing your voice..."):

            audio_path = None

            try:

                whisper = load_whisper()

                with tempfile.NamedTemporaryFile(
                    suffix=".wav",
                    delete=False
                ) as temp_audio:

                    temp_audio.write(audio_input.getvalue())
                    audio_path = temp_audio.name

                segments, info = whisper.transcribe(
                    audio_path,
                    beam_size=5,
                    vad_filter=True
                )

                recognized_text = " ".join(
                    segment.text for segment in segments
                ).strip()

                if not recognized_text:

                    st.error(
                        "Voice not recognized. Please try again."
                    )
                    st.stop()

                final_question = recognized_text

            except Exception as e:

                st.error(f"Voice recognition error: {e}")
                st.stop()

            finally:

                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)

    # -----------------------
    # VALIDATE QUESTION
    # -----------------------

    if not final_question:

        st.warning("Please type or speak your question.")
        st.stop()

    st.session_state.current_question = final_question

    db = st.session_state.db
    llm = st.session_state.llm

    # -----------------------
    # TRANSLATE TO ENGLISH
    # -----------------------

    with st.spinner("🌐 Understanding your question..."):

        try:

            english_question = translate_question(
                llm,
                final_question
            )

        except Exception as e:

            st.error(f"Question translation error: {e}")
            st.stop()

    # -----------------------
    # RETRIEVE DOCUMENTS
    # -----------------------

    with st.spinner("📄 Searching Oncology Documents..."):

        try:

            docs = db.similarity_search(
                english_question,
                k=5
            )

            st.session_state.docs = docs

        except Exception as e:

            st.error(f"Retrieval error: {e}")
            st.stop()

    if not docs:

        st.warning("No relevant documents found.")
        st.stop()

    # -----------------------
    # BUILD CONTEXT
    # -----------------------

    context_parts = []

    for i, doc in enumerate(docs, start=1):

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"""
Source {i}
PDF: {source}
Page: {page}

Content:
{doc.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    # -----------------------
    # GENERATE ANSWER
    # -----------------------

    with st.spinner("🤖 Generating Multilingual Answer..."):

        try:

            answer = generate_answer(
                llm,
                context,
                english_question,
                selected_language
            )

            st.session_state.answer = answer

        except Exception as e:

            st.error(f"Answer generation error: {e}")
            st.stop()

    # -----------------------
    # AUTOMATIC TEXT TO SPEECH
    # -----------------------

    with st.spinner("🔊 Preparing Voice Answer..."):

        try:

            speech = gTTS(
                text=answer,
                lang=LANGUAGES[selected_language]["code"],
                slow=False
            )

            audio_buffer = BytesIO()

            speech.write_to_fp(audio_buffer)

            st.session_state.answer_audio = (
                audio_buffer.getvalue()
            )

        except Exception as e:

            st.session_state.answer_audio = None

            st.warning(
                f"Text answer generated, but voice generation failed: {e}"
            )


# ---------------------------
# DISPLAY QUESTION
# ---------------------------

if st.session_state.current_question:

    st.subheader("🎤 Your Question")

    st.write(st.session_state.current_question)


# ---------------------------
# DISPLAY ANSWER
# ---------------------------

if st.session_state.answer:

    st.subheader("🤖 Answer")

    st.markdown(st.session_state.answer)

    st.subheader("🔊 Listen to Answer")

    if st.session_state.answer_audio:

        st.audio(
            st.session_state.answer_audio,
            format="audio/mp3",
            autoplay=True
        )

    else:

        st.info("Voice is not available for this answer.")


# ---------------------------
# RETRIEVED SOURCES
# ---------------------------

if st.session_state.answer and st.session_state.docs:

    st.subheader("📄 Retrieved Sources")

    for i, doc in enumerate(
        st.session_state.docs,
        start=1
    ):

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "?"
        )

        st.markdown(
            f"""
**Source {i}**

- PDF: {source}
- Page: {page}
"""
        )


# ---------------------------
# RETRIEVED CHUNKS
# ---------------------------

if st.session_state.answer and st.session_state.docs:

    with st.expander("🔍 View Retrieved Chunks"):

        for i, doc in enumerate(
            st.session_state.docs,
            start=1
        ):

            st.markdown(f"### Chunk {i}")

            st.write(doc.page_content)

            st.divider()


# ---------------------------
# MEDICAL DISCLAIMER
# ---------------------------

st.caption(
    "This system is for educational and research purposes only. "
    "It is not a substitute for professional medical advice."
)
