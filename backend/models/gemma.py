from __future__ import annotations
import os
import time
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModel

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)

class ModelLoader:
    _instance: Optional["ModelLoader"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
        show_timing: bool = False,
    ):
        if getattr(self, "initialized", False):
            return
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device.startswith("cuda") else torch.float32)
        self.trust_remote_code = trust_remote_code
        self.tokenizer = None
        self.model = None
        self.initialized = False
        self._load(show_timing)
        self.initialized = True

    def _load(self, show_timing: bool):
        kwargs = {"trust_remote_code": self.trust_remote_code}
        if HF_TOKEN:
            kwargs["use_auth_token"] = HF_TOKEN
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, **kwargs)
        if show_timing:
            print("tokenizer:", time.time() - t0)
        t1 = time.time()
        model_kwargs = {"dtype": self.dtype, "trust_remote_code": self.trust_remote_code}
        if HF_TOKEN:
            model_kwargs["use_auth_token"] = HF_TOKEN
        self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs).to(self.device)
        self.model.eval()
        if show_timing:
            print("model:", time.time() - t1)
