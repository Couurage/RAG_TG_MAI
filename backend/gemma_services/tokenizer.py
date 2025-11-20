from transformers import AutoTokenizer
from pathlib import Path
from typing import List, Sequence, Optional, Dict, Any

class Tokenizer:
    def __init__(
        self,
        model_name: str = "unsloth/embeddinggemma-300m-qat-q8_0-unquantized",
        max_tokens: int = 250,
        overlap: int = 50,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if overlap >= max_tokens:
            raise ValueError("overlap must be smaller than max_tokens to avoid infinite loops")
        self.max_tokens = max_tokens
        self.overlap = overlap

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _md_to_blocks(self, md: str) -> List[str]:
        return [b.strip() for b in md.split("\n\n") if b.strip()]

    def _split_long_block(self, text: str) -> List[str]:
        toks = self.tokenizer.encode(text, add_special_tokens=False)
        res = []
        start = 0
        while start < len(toks):
            end = start + self.max_tokens
            decoded = self.tokenizer.decode(
                toks[start:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            chunk = decoded.strip()
            res.append(chunk)
            start += self.max_tokens - self.overlap
        return res

    def chunk_markdown(self, md: str) -> List[str]:
        blocks = self._md_to_blocks(md)
        chunks: List[str] = []
        current = ""
        for block in blocks:
            candidate = (current + "\n\n" + block).strip() if current else block
            if self.count_tokens(candidate) <= self.max_tokens:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if self.count_tokens(block) > self.max_tokens:
                    chunks.extend(self._split_long_block(block))
                    current = ""
                else:
                    current = block
        if current:
            chunks.append(current)
        return chunks

    def chunk_folder(self, folder: str | Path) -> dict[str, List[str]]:
        folder = Path(folder)
        result: dict[str, List[str]] = {}
        for file in folder.glob("*.md"):
            text = file.read_text(encoding="utf-8")
            result[file.name] = self.chunk_markdown(text)
        return result
