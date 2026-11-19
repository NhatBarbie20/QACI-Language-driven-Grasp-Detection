"""Frozen CLIP text embeddings for grasp instructions.

The text encoder is never trained and the dataset has only a few thousand
distinct instructions, so every instruction is encoded once and stored in
``<data root>/text_embeddings.pt``. Training then needs neither the CLIP model
nor a tokenizer, which keeps GPU memory and epoch time low.
"""

from __future__ import annotations

from pathlib import Path

import torch

from lgrasp.data.manifest import load_manifest

EMBEDDINGS_NAME = "text_embeddings.pt"
DEFAULT_TEXT_MODEL = "openai/clip-vit-base-patch32"


@torch.no_grad()
def encode_texts(texts: list[str], model_name: str = DEFAULT_TEXT_MODEL, batch_size: int = 256,
                 device: str | None = None) -> torch.Tensor:
    """Encode ``texts`` with the CLIP text tower; returns L2-normalised ``[N, D]`` float32."""
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer  # heavy import, only needed here

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device).eval()
    chunks = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[start : start + batch_size], padding=True, truncation=True, return_tensors="pt")
        embeddings = model(**tokens.to(device)).text_embeds
        chunks.append(torch.nn.functional.normalize(embeddings.float(), dim=-1).cpu())
    return torch.cat(chunks)


def build_text_embeddings(root: str | Path, model_name: str = DEFAULT_TEXT_MODEL) -> Path:
    """Encode every instruction of the manifest and save the lookup table."""
    root = Path(root)
    texts = sorted({sample["instruction"] for sample in load_manifest(root)["samples"]})
    path = root / EMBEDDINGS_NAME
    torch.save({"model": model_name, "texts": texts, "embeddings": encode_texts(texts, model_name)}, path)
    return path


class TextEmbeddings:
    """Lookup table from instruction string to its embedding."""

    def __init__(self, root: str | Path):
        path = Path(root) / EMBEDDINGS_NAME
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python scripts/encode_text.py --root {root}")
        table = torch.load(path, map_location="cpu", weights_only=True)
        self.model_name: str = table["model"]
        self.embeddings: torch.Tensor = table["embeddings"]
        self._index = {text: position for position, text in enumerate(table["texts"])}

    @property
    def dim(self) -> int:
        return self.embeddings.shape[1]

    def __call__(self, texts: list[str]) -> torch.Tensor:
        """Embeddings ``[len(texts), D]`` in the given order."""
        return self.embeddings[[self._index[text] for text in texts]]
