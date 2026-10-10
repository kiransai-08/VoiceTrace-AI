from faster_whisper import WhisperModel

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

segments, info = model.transcribe("audio/test.wav")

print("\nTranscript:")

for segment in segments:
    print(segment.text)