import streamlit as st
from faster_whisper import WhisperModel
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)
import torch
import tempfile
import os


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Humanized Detector",
    page_icon="🎤",
    layout="centered"
)

st.title("🎤 AI Humanized Text Detector")

st.write(
    "Record your voice, convert it to text, "
    "and analyze whether the transcript is AI-generated or human-written."
)


# ============================================================
# 1. LOAD WHISPER
# ============================================================

@st.cache_resource
def load_whisper():

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

    return model


# ============================================================
# 2. LOAD DISTILBERT
# ============================================================

@st.cache_resource
def load_detector():

    model_path = "./models/three_class_detector"

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        model_path
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        model_path
    )

    model.eval()

    return tokenizer, model


whisper_model = load_whisper()
tokenizer, detector_model = load_detector()


# ============================================================
# 3. RECORD VOICE
# ============================================================

audio = st.audio_input("🎙️ Record your voice")


if audio is not None:

    st.subheader("🎵 Recorded Audio")

    st.audio(audio)


    # ========================================================
    # SAVE AUDIO TEMPORARILY
    # ========================================================

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_file:

        temp_file.write(audio.getvalue())

        audio_path = temp_file.name


    try:

        # ====================================================
        # 4. SPEECH → TEXT
        # ====================================================

        st.subheader("📝 Transcript")

        segments, info = whisper_model.transcribe(
            audio_path
        )

        transcript = ""

        for segment in segments:

            transcript += segment.text + " "

        transcript = transcript.strip()


        if transcript:

            st.success("Speech converted successfully!")

            st.text_area(
                "Your Speech",
                transcript,
                height=150
            )


            # ================================================
            # 5. TEXT → AI/HUMAN CLASSIFICATION
            # ================================================

            st.subheader("🤖 AI Detection")

            inputs = tokenizer(
                transcript,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )


            with torch.no_grad():

                outputs = detector_model(
                    **inputs
                )

                probabilities = torch.softmax(
                    outputs.logits,
                    dim=1
                )[0]


            # ================================================
            # 6. GET PROBABILITIES
            # ================================================

            human_probability = (
                probabilities[0].item() * 100
            )

            ai_probability = (
                probabilities[1].item() * 100
            )

            humanised_probability = (
                probabilities[2].item() * 100
            )


            # ================================================
            # 7. DISPLAY RESULTS
            # ================================================

            st.markdown("### 📊 Result")

            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "👤 Human",
                    f"{human_probability:.2f}%"
                )


            with col2:

                st.metric(
                    "🤖 AI",
                    f"{ai_probability:.2f}%"
                )


            with col3:

                st.metric(
                    "🎭 AI-Humanised",
                    f"{humanised_probability:.2f}%"
                )


            # ================================================
            # 8. FINAL PREDICTION
            # ================================================

            max_prob_idx = torch.argmax(probabilities).item()

            if max_prob_idx == 0:

                st.success(
                    "👤 Prediction: Human Written"
                )

            elif max_prob_idx == 1:

                st.error(
                    "🤖 Prediction: AI Generated"
                )

            else:

                st.warning(
                    "🎭 Prediction: AI-Humanised"
                )


        else:

            st.warning(
                "No speech detected."
            )


    except Exception as e:

        st.error(
            f"Error: {e}"
        )


    finally:

        if os.path.exists(audio_path):

            os.remove(audio_path)