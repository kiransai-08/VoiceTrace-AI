# 🎤 VoiceTrace AI (AI Humanized Text Detector)

VoiceTrace AI is a real-time, voice-based artificial intelligence text detection system. It allows users to record their voice, transcribes the speech to text, and uses a fine-tuned Transformer model to classify whether the transcript is **Human-written**, **AI-generated**, or **AI-Humanised** (AI text edited to sound more human).

## Features

- **Speech-to-Text**: Fast and accurate transcription using [Faster Whisper](https://github.com/guillaumekln/faster-whisper).
- **3-Class Transformer Classification**: Uses a fine-tuned DistilBERT model to classify text into:
  - `Human`
  - `AI`
  - `AI-Humanised`
- **Interactive UI**: Built with [Streamlit](https://streamlit.io/) for an intuitive and seamless user experience.

## Pipeline Architecture

1. **Audio Input**: User speaks into the microphone via the Streamlit interface.
2. **Transcription**: The audio is processed by `faster-whisper` running locally to generate a transcript.
3. **Classification**: The transcript is fed into our fine-tuned 3-class DistilBERT sequence classification model.
4. **Results**: The system outputs probabilities for all three classes and displays the final prediction.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/kiransai-08/VoiceTrace-AI.git
cd VoiceTrace-AI
```

### 2. Set up a virtual environment (Optional but recommended)
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
Install the required packages using `pip`:
```bash
pip install -r requirment.txt
```

*(Note: Depending on your system, you may also need to install PyTorch with CUDA support if you wish to run the models on a GPU).*

### 4. Download / Train the Model
By default, the application expects the fine-tuned 3-class model to be located in the `./models/three_class_detector` directory.

If you are training from scratch, you can use the provided data preparation and training scripts (e.g., `prepare_3class_dataset.py`, `train_three_class.py`) to fetch the dataset and train the model.

## Usage

Start the Streamlit web application by running:

```bash
streamlit run app.py
```

This will launch the app in your default web browser (typically at `http://localhost:8501`). Record your voice and click to analyze!

## Acknowledgements

- **Dataset**: Built using a subset of the [HC3 Human AI Reborn Text Dataset](https://huggingface.co/datasets) to include the difficult `AI-Humanised` examples.
- **Models**: Powered by Hugging Face `transformers` and `faster-whisper`.
