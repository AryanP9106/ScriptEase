"""Extract text from plain text, DOCX, and text-based PDF uploads."""

from __future__ import annotations

import io
import zipfile
from xml.etree import ElementTree

import pdfplumber


def extract_text(filename: str, data: bytes) -> str:
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension == "txt":
        return data.decode("utf-8", errors="replace")
    if extension == "docx":
        return _extract_docx(data)
    if extension == "pdf":
        return _extract_pdf(data)
    raise ValueError("Supported file types are TXT, DOCX, and text-based PDF.")


def _extract_docx(data: bytes) -> str:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(data)) as document:
        root = ElementTree.fromstring(document.read("word/document.xml"))
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace))
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _extract_pdf(data: bytes) -> str:
    with pdfplumber.open(io.BytesIO(data)) as document:
        return "\n".join(page.extract_text() or "" for page in document.pages)
