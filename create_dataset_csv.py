import os
import pandas as pd

audio_dir = "speech_dataset/audio"
text_dir = "speech_dataset/transcripts"

data = []

for file in os.listdir(audio_dir):
    if file.endswith(".mp3"):
        audio_path = os.path.join(audio_dir, file)
        text_file = file.replace(".mp3", ".txt")
        text_path = os.path.join(text_dir, text_file)

        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            data.append({
                "audio_path": audio_path,
                "text": text
            })

df = pd.DataFrame(data)
df.to_csv("speech_dataset.csv", index=False)

print("✅ CSV created successfully!")