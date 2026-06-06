import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

VIDEO_ID = "Rni7Fz7208c"

with open("transcript.json", "r", encoding="utf-8") as f:
    transcript = json.load(f)

index = faiss.read_index(
    "faiss_index.bin"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

while True:
    question = input("\nAsk a question (or type exit): ")

    if question.lower() == "exit":
        break

    q_embedding = model.encode([question])

    D, I = index.search(
        np.array(q_embedding).astype("float32"),
        1
    )

    answer_segment = transcript[I[0][0]]

    start_time = int(answer_segment["start"])

    youtube_link = (
        f"https://www.youtube.com/watch?v={VIDEO_ID}&t={start_time}s"
    )

    print("\nAnswer:")
    print(answer_segment["text"])

    print("\nTimestamp:")
    print(start_time)

    print("\nYouTube Link:")
    print(youtube_link)
