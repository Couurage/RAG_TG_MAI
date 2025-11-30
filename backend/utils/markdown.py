from __future__ import annotations
import hashlib
import logging
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from markitdown import MarkItDown
from markitdown._exceptions import FileConversionException

log = logging.getLogger(__name__)

def _safe_stem(path: Path) -> str:
    # стабильное имя по хэшу исходника (чтобы не плодить дубликаты)
    h = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    return f"{path.stem}-{h}"

_DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _docx_fallback_text(path: Path) -> str:
    """
    Simple docx -> text fallback when MarkItDown fails on malformed styles.
    Extracts paragraph text from the document XML to avoid brittle style parsing.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            with zf.open("word/document.xml") as fh:
                root = ET.fromstring(fh.read())
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise RuntimeError(f"Failed to read docx XML: {exc}") from exc

    paragraphs: list[str] = []
    for p in root.findall(".//w:p", namespaces=_DOCX_NS):
        texts: list[str] = []
        for node in p.findall(".//w:t", namespaces=_DOCX_NS):
            if node.text:
                texts.append(node.text)
        paragraph = "".join(texts).strip()
        if paragraph:
            paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)


def convert_to_md(
    input_path: str | Path,
    out_dir: str | Path = "data_md",
    extract_images: bool = False,
) -> tuple[Path, int]:
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    try:
        result = md.convert(str(in_path))
        md_text = result.text_content or ""
    except FileConversionException as exc:
        if in_path.suffix.lower() == ".docx":
            log.warning("MarkItDown failed for %s, using docx fallback: %s", in_path, exc)
            md_text = _docx_fallback_text(in_path)
        else:
            raise

    full_hash = hashlib.sha1(in_path.read_bytes()).hexdigest()
    short_hash = full_hash[:10]

    base = f"{in_path.stem}-{short_hash}"
    md_path = out_dir / f"{base}.md"
    md_text = md_text.replace("\r\n", "\n").strip() + "\n"

    md_path.write_text(md_text, encoding="utf-8")
    log.info("Saved: %s", md_path)

    doc_id = int(short_hash, 16)
    return md_path, doc_id
