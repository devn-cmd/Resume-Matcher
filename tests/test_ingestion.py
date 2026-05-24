from pathlib import Path
from src.ingestion import read_document

def test_reads_txt(tmp_path):
    p = tmp_path / "r.txt"
    p.write_text("Python developer")
    assert "Python" in read_document(p)