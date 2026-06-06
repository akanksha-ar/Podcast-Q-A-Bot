import warnings
warnings.filterwarnings("ignore")

import whisper
import json
import os

model = whisper.load_model("base")

audio_file = "audio.wav"
if not os.path.exists(audio_file):
    print(f"Error: {audio_file} not found!")
    exit()

print("Transcribing... please wait.")

result = model.transcribe(
    audio_file,
    language="en",           # Force English
    word_timestamps=True,
    verbose=False,
    condition_on_previous_text=False,  # Prevents looping garbage text
    temperature=0.0           # More deterministic output
)

transcript = []

for segment in result["segments"]:
    # Skip garbage/repeated segments
    text = segment["text"].strip()
    if not text or len(set(text.split())) < 3:
        continue

    entry = {
        "start": round(segment["start"], 2),
        "end": round(segment["end"], 2),
        "text": text
    }

    if "words" in segment:
        entry["words"] = [
            {
                "word": w["word"].strip(),
                "start": round(w["start"], 2),
                "end": round(w["end"], 2)
            }
            for w in segment["words"]
        ]

    transcript.append(entry)

with open("transcript.json", "w", encoding="utf-8") as f:
    json.dump(transcript, f, indent=4, ensure_ascii=False)

print(f"✅ Transcript saved to transcript.json")
print(f"   Total segments: {len(transcript)}")
print(result["text"][:500])
