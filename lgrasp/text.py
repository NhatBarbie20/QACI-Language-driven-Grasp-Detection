"""Frozen CLIP text embeddings for grasp instructions.

The text encoder is never trained and the dataset has only a few thousand
distinct instructions, so every instruction is encoded once and stored in
``<data root>/text_embeddings_<clip model>.pt``. Training then needs neither the CLIP model
nor a tokenizer, which keeps GPU memory and epoch time low.
"""

from __future__ import annotations

from pathlib import Path

import torch

from lgrasp.data.manifest import load_manifest

DEFAULT_TEXT_MODEL = "openai/clip-vit-base-patch32"
PROMPTS = ("sentence", "noun")
# CLIP is trained on captions, not commands: its recommended zero-shot prompt.
NOUN_TEMPLATE = "a photo of a {}"


def embeddings_path(root: str | Path, model_name: str) -> Path:
    """Cache file of a CLIP checkpoint, e.g. ``text_embeddings_clip-vit-base-patch32.pt``."""
    return Path(root) / f"text_embeddings_{model_name.split('/')[-1]}.pt"


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
    samples = load_manifest(root)["samples"]
    texts = {sample["instruction"] for sample in samples}
    texts.update(NOUN_TEMPLATE.format(sample["object_name"]) for sample in samples)
    texts = sorted(texts)
    path = embeddings_path(root, model_name)
    torch.save({"model": model_name, "texts": texts, "embeddings": encode_texts(texts, model_name)}, path)
    return path


class TextEmbeddings:
    """Lookup table from instruction string to its embedding.

    With ``prompt="noun"`` an instruction is represented by the object it names,
    encoded as :data:`NOUN_TEMPLATE`, instead of by the full sentence.
    """

    def __init__(self, root: str | Path, model_name: str = DEFAULT_TEXT_MODEL, prompt: str = "sentence"):
        if prompt not in PROMPTS:
            raise ValueError(f"Unknown prompt {prompt!r}; expected one of {PROMPTS}")
        path = embeddings_path(root, model_name)
        if not path.exists():  # first use of this checkpoint: encode once, then reuse
            build_text_embeddings(root, model_name)
        table = torch.load(path, map_location="cpu", weights_only=True)
        self._alias: dict[str, str] = {}
        if prompt == "noun":
            self._alias = {
                sample["instruction"]: NOUN_TEMPLATE.format(sample["object_name"])
                for sample in load_manifest(root)["samples"]
            }
            if not set(self._alias.values()) <= set(table["texts"]):  # table from before noun prompts
                build_text_embeddings(root, model_name)
                table = torch.load(path, map_location="cpu", weights_only=True)
        self.model_name: str = table["model"]
        self.embeddings: torch.Tensor = table["embeddings"]
        self._index = {text: position for position, text in enumerate(table["texts"])}

    @property
    def dim(self) -> int:
        return self.embeddings.shape[1]

    def __call__(self, texts: list[str]) -> torch.Tensor:
        """Embeddings ``[len(texts), D]`` in the given order."""
        return self.embeddings[[self._index[self._alias.get(text, text)] for text in texts]]
