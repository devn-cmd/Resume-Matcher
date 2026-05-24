# src/ingestion.py
from pathlib import Path
import pdfplumber
import docx


def _read_pdf(path: Path) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _read_docx(path: Path) -> str:
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)


def _read_txt(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


READERS = {".pdf": _read_pdf, ".docx": _read_docx, ".txt": _read_txt}


def read_document(path) -> str:
    """Extract raw text from a PDF, DOCX, or TXT file."""
    path = Path(path)
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported format: {path.suffix}")
    return reader(path).strip()


def read_folder(folder) -> dict[str, str]:
    """Read every supported file in a folder -> {filename: text}."""
    folder = Path(folder)
    out = {}
    for f in folder.iterdir():
        if f.suffix.lower() in READERS:
            out[f.name] = read_document(f)
    return out