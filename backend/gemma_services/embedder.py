from __future__ import annotations
from typing import List, Optional, Sequence
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class Embedder:
    def __init__(
        self,
        model_name: str = "unsloth/embeddinggemma-300m-qat-q8_0-unquantized",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        max_length: int = 2048,
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() elif "mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" or self.device == "mps" else torch.float32)
        self.max_length = max_length

        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()
        self._dim_cache = None

    @torch.no_grad()
    def embed(self, texts: str | Sequence[str], batch_size: int = 16) -> List[List[float]]:
        if isinstance(texts, str):
            batch_texts: List[str] = [texts]
        else:
            batch_texts = list(texts)

        if not batch_texts:
            return []

        out: List[List[float]] = []
        for i in range(0, len(batch_texts), batch_size):
            batch = batch_texts[i : i + batch_size]
            toks = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**toks)
            hidden = outputs.last_hidden_state
            mask = toks.attention_mask.unsqueeze(-1)
            masked = hidden * mask
            summed = masked.sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            mean_pooled = summed / counts

            normed = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            out.extend(normed.float().cpu().tolist())

            if self._dim_cache is None:
                self._dim_cache = normed.shape[-1]

        return out

    def dim(self) -> int:
        if self._dim_cache is not None:
            return self._dim_cache
        v = self.embed(["probe"], batch_size=1)[0]
        self._dim_cache = len(v)
        return self._dim_cache
