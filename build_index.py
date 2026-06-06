import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

with open("transcript.json", "r", encoding="utf-8") as f:
    transcript = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [seg["text"] for seg in transcript]

embeddings = model.encode(texts)

index = faiss.IndexFlatL2(
    embeddings.shape[1]
)

index.add(
    np.array(embeddings).astype("float32")
)

faiss.write_index(
    index,
    "faiss_index.bin"
)

print("FAISS index saved!")
