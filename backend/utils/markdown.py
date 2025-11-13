from __future__ import annotations
import hashlib
import logging
from pathlib import Path
from typing import Optional

from markitdown import MarkItDown

log = logging.getLogger(__name__)

def _safe_stem(path: Path) -> str:
    # стабильное имя по хэшу исходника (чтобы не плодить дубликаты)
    h = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    return f"{path.stem}-{h}"

def convert_to_md(
    input_path: str | Path,
    out_dir: str | Path = "data_md",
    extract_images: bool = False,
) -> Path:

    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    result = md.convert(str(in_path))

    base = _safe_stem(in_path)
    md_path = out_dir / f"{base}.md"
    md_text = result.text_content or ""

    # грубая постобработка
    md_text = md_text.replace("\r\n", "\n").strip() + "\n"

    md_path.write_text(md_text, encoding="utf-8")
    log.info("Saved: %s", md_path)
    return md_path
