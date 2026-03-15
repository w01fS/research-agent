import os
import json
import faiss
import numpy as np


class VectorMemory:

    def __init__(self, dim: int = 384, storage_dir: str = "data"):
        self.dim = dim
        self.storage_dir = storage_dir
        self.index_path = os.path.join(storage_dir, "memory.faiss")
        self.docs_path = os.path.join(storage_dir, "memory.json")

        # Load from disk if exists, otherwise create fresh
        if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.docs_path, "r") as f:
                self.documents = json.load(f)
            print(f"  [MEMORY] Loaded {len(self.documents)} memories from disk")
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.documents = []

    def add(self, embedding, text):
        if text in self.documents:
            return
        vector = np.array([embedding]).astype("float32")
        self.index.add(vector)
        self.documents.append(text)
        self._save()

    def search(self, embedding, k=3):
        if len(self.documents) == 0:
            return []

        vector = np.array([embedding]).astype("float32")
        distances, indices = self.index.search(vector, k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])

        return results

    def _save(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.docs_path, "w") as f:
            json.dump(self.documents, f, indent=2)