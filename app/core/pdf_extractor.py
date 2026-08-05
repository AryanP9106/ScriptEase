"""PDF-to-image extraction utilities for the ScriptEase pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from pdf2image import convert_from_path
from tqdm import tqdm


class PDFExtractor:
    """Extract every page of a PDF as a numbered PNG image."""

    def __init__(self, poppler_path: Optional[Union[str, Path]] = None) -> None:
        self.poppler_path = Path(poppler_path) if poppler_path else None

    def extract(
        self,
        pdf_path: Union[str, Path],
        output_folder: Union[str, Path],
        dpi: int = 300,
    ) -> list[Path]:
        """Extract pages from `pdf_path` into `output_folder` and return them."""
        pdf_path = Path(pdf_path)
        output_folder = Path(output_folder)

        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if dpi <= 0:
            raise ValueError("dpi must be greater than zero")

        output_folder.mkdir(parents=True, exist_ok=True)
        kwargs = {"poppler_path": str(self.poppler_path)} if self.poppler_path else {}
        pages = convert_from_path(str(pdf_path), dpi=dpi, **kwargs)

        written_paths: list[Path] = []
        for index, page in enumerate(tqdm(pages, desc="Extracting"), start=1):
            output_path = output_folder / f"page_{index:03}.png"
            page.save(output_path, "PNG")
            written_paths.append(output_path)

        return written_paths
