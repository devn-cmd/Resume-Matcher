import numpy as np
from sentence_transformers import SentenceTransformer
import config

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.MODEL_NAME)
    return _model


def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
    model = get_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,   # dot product == cosine similarity
        show_progress_bar=False,
    )


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]