# extract_pdfs.py  →  RUN THIS NOW
import fitz
import os
from pathlib import Path

pdf_folder = Path("pdfs")
output = Path("data/fast_edu_notes")
output.mkdir(parents=True, exist_ok=True)


for pdf in pdf_folder.glob("*.pdf"):
    try:
        doc = fitz.open(pdf)
        text = ""
        for page in doc:
            text += page.get_text()
        (output / f"{pdf.stem}.txt").write_text(text, encoding="utf-8")
        print(f"Extracted → {pdf.name}")
    except:
        print(f"Failed → {pdf.name}")