"""ScriptEase personal handwriting PDF generator.

Run with: streamlit run main.py
"""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from app.core.handwriting import HandwritingLibrary, HandwritingRenderer, RenderOptions, save_pdf
from app.core.personal_handwriting import CHARACTERS, extract_page, extract_pdf_document, load_glyphs, save_glyphs, sheet_cells, sheets_as_pdf
from app.core.text_extractor import extract_text


GLYPH_DIR = Path("dataset/personal/glyphs")


def build_pdf(text: str, size: int, color: str) -> tuple[bytes, bytes, int]:
    glyphs = load_glyphs(GLYPH_DIR)
    if not glyphs:
        raise ValueError("Create your personal glyph library first.")

    rgb = bytes.fromhex(color.lstrip("#"))
    renderer = HandwritingRenderer(
        HandwritingLibrary(glyphs),
        RenderOptions(
            font_size=size,
            ink_color=(rgb[0], rgb[1], rgb[2]),
        ),
    )
    pages = renderer.render(text)
    with TemporaryDirectory() as directory:
        pdf_path = Path(directory) / "handwritten.pdf"
        preview_path = Path(directory) / "preview.png"
        save_pdf(pages, pdf_path)
        pages[0].save(preview_path)
        return pdf_path.read_bytes(), preview_path.read_bytes(), len(pages)


st.set_page_config(page_title="ScriptEase", page_icon="✍️", layout="wide")
st.title("ScriptEase")
st.caption("Create typed PDFs in your actual handwriting—without OCR-based character guessing.")

sheet_tab, capture_tab, write_tab = st.tabs(["1. Writing sheets", "2. Build handwriting", "3. Create PDF"])

with sheet_tab:
    st.subheader("Print and fill the sheets")
    st.write("Each box is labelled. Write the exact character once in the inner rectangle. The three copies create natural variation.")
    st.download_button("Download personal handwriting sheets", sheets_as_pdf(), "scriptease-writing-sheets.pdf", "application/pdf", type="primary")
    st.info("Print at 100% scale. Scan each page flat and uncropped, with all four black corner markers visible.")

with capture_tab:
    st.subheader("Upload completed sheets")
    expected_pages = len(sheet_cells())
    uploaded_sheets = st.file_uploader(
        f"Upload one {expected_pages}-page PDF or all {expected_pages} sheet images in page order", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True
    )
    if st.button("Build personal glyph library", type="primary"):
        is_pdf = len(uploaded_sheets) == 1 and uploaded_sheets[0].name.lower().endswith(".pdf")
        if not is_pdf and len(uploaded_sheets) != expected_pages:
            st.error(f"Upload one {expected_pages}-page PDF or exactly {expected_pages} completed sheet images in page order.")
        else:
            glyphs: dict[str, list] = {}
            try:
                with st.spinner("Aligning pages and extracting known character boxes…"):
                    if is_pdf:
                        glyphs = extract_pdf_document(uploaded_sheets[0].getvalue())
                    else:
                        for page_number, sheet in enumerate(uploaded_sheets, start=1):
                            for character, images in extract_page(sheet.getvalue(), page_number).items():
                                glyphs.setdefault(character, []).extend(images)
                if GLYPH_DIR.exists():
                    shutil.rmtree(GLYPH_DIR)
                save_glyphs(glyphs, GLYPH_DIR)
            except Exception as error:
                st.error(f"Could not build the library: {error}")
            else:
                missing = sorted(set(CHARACTERS) - set(glyphs))
                if missing:
                    st.warning(f"Saved {sum(map(len, glyphs.values()))} glyphs. Missing boxes: {''.join(missing)}")
                else:
                    st.success(f"Saved {sum(map(len, glyphs.values()))} personal glyphs.")

with write_tab:
    st.subheader("Turn typed content into your handwritten PDF")
    glyph_count = sum(len(images) for images in load_glyphs(GLYPH_DIR).values()) if GLYPH_DIR.exists() else 0
    st.caption(f"Personal glyphs available: {glyph_count}")
    uploaded_document = st.file_uploader("Upload typed TXT, DOCX, or text-based PDF", type=["txt", "docx", "pdf"])
    text = st.text_area("Or paste typed text", height=220)
    if uploaded_document is not None:
        try:
            text = extract_text(uploaded_document.name, uploaded_document.getvalue())
            st.text_area("Extracted text", text, height=180, disabled=True)
        except Exception as error:
            st.error(f"Could not read the document: {error}")
    size = st.slider("Writing size", 20, 46, 30)
    color = st.color_picker("Ink colour", "#142746")
    if st.button("Create handwritten PDF", type="primary"):
        try:
            with st.spinner("Writing your document…"):
                pdf_data, preview_data, page_count = build_pdf(text, size, color)
        except Exception as error:
            st.error(f"Could not create the document: {error}")
        else:
            st.success(f"Created {page_count} page(s).")
            st.image(preview_data, caption="First-page preview", use_container_width=True)
            st.download_button("Download handwritten PDF", pdf_data, "handwritten.pdf", "application/pdf", type="primary")
