"""Create aligned handwriting sheets and build a personal glyph library."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont


CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?'-()[]{}@#&%+=*/_"
PAGE_SIZE = (2480, 3508)
COLUMNS, ROWS = 5, 10
MARKER_SIZE = 90


@dataclass(frozen=True)
class Cell:
    character: str
    variant: int
    bounds: tuple[int, int, int, int]


def sheet_cells() -> list[list[Cell]]:
    slots = [(character, variant) for variant in range(3) for character in CHARACTERS]
    left, top, right, bottom = 110, 170, PAGE_SIZE[0] - 110, PAGE_SIZE[1] - 150
    cell_width = (right - left) // COLUMNS
    cell_height = (bottom - top) // ROWS
    pages: list[list[Cell]] = []
    for page_start in range(0, len(slots), COLUMNS * ROWS):
        page: list[Cell] = []
        for index, (character, variant) in enumerate(slots[page_start : page_start + COLUMNS * ROWS]):
            column, row = index % COLUMNS, index // COLUMNS
            x1, y1 = left + column * cell_width, top + row * cell_height
            page.append(Cell(character, variant, (x1, y1, x1 + cell_width, y1 + cell_height)))
        pages.append(page)
    return pages


def create_sheets() -> list[Image.Image]:
    """Create printable A4 worksheet pages with fixed character locations."""
    font = ImageFont.truetype("arial.ttf", 34)
    small_font = ImageFont.truetype("arial.ttf", 22)
    pages = []
    all_pages = sheet_cells()
    for page_number, cells in enumerate(all_pages, start=1):
        page = Image.new("RGB", PAGE_SIZE, "white")
        draw = ImageDraw.Draw(page)
        for x, y in ((20, 20), (PAGE_SIZE[0] - MARKER_SIZE - 20, 20), (20, PAGE_SIZE[1] - MARKER_SIZE - 20), (PAGE_SIZE[0] - MARKER_SIZE - 20, PAGE_SIZE[1] - MARKER_SIZE - 20)):
            draw.rectangle((x, y, x + MARKER_SIZE, y + MARKER_SIZE), fill="black")
        draw.text((PAGE_SIZE[0] // 2, 35), f"ScriptEase character sheet {page_number}/{len(all_pages)}", fill="black", anchor="ma", font=font)
        draw.text((PAGE_SIZE[0] // 2, 85), "Write one clear character in each box. Keep ink inside the inner area.", fill="black", anchor="ma", font=small_font)
        for cell in cells:
            x1, y1, x2, y2 = cell.bounds
            draw.rectangle(cell.bounds, outline="black", width=2)
            draw.text((x1 + 16, y1 + 12), f"{cell.character}  ({cell.variant + 1}/3)", fill="black", font=small_font)
            draw.rectangle((x1 + 25, y1 + 55, x2 - 25, y2 - 25), outline=(190, 190, 190), width=1)
        pages.append(page)
    return pages


def sheets_as_pdf() -> bytes:
    pages = create_sheets()
    output = io.BytesIO()
    pages[0].save(output, "PDF", save_all=True, append_images=pages[1:], resolution=300.0)
    return output.getvalue()


def _align(image: Image.Image) -> np.ndarray:
    source = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)[1]
    height, width = binary.shape
    centres: list[tuple[float, float] | None] = []
    for x_start, y_start, x_end, y_end in ((0, 0, width // 4, height // 3), (width * 3 // 4, 0, width, height // 3), (0, height * 2 // 3, width // 4, height), (width * 3 // 4, height * 2 // 3, width, height)):
        region = binary[y_start:y_end, x_start:x_end]
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = [cv2.boundingRect(contour) for contour in contours]
        candidates = [rect for rect in candidates if rect[2] >= 20 and rect[3] >= 20 and 0.5 <= rect[2] / rect[3] <= 2.0]
        if not candidates:
            centres.append(None)
            continue
        corner_x = x_start if x_start == 0 else x_end
        corner_y = y_start if y_start == 0 else y_end
        x, y, w, h = min(
            candidates,
            key=lambda rect: (x_start + rect[0] + rect[2] / 2 - corner_x) ** 2
            + (y_start + rect[1] + rect[3] / 2 - corner_y) ** 2,
        )
        centres.append((x_start + x + w / 2, y_start + y + h / 2))
    target = np.float32(((65, 65), (PAGE_SIZE[0] - 65, 65), (65, PAGE_SIZE[1] - 65), (PAGE_SIZE[0] - 65, PAGE_SIZE[1] - 65)))
    if all(centre is not None for centre in centres):
        matrix = cv2.getPerspectiveTransform(np.float32(centres), target)
        return cv2.warpPerspective(source, matrix, PAGE_SIZE, borderValue=(255, 255, 255))
    if centres[0] is not None and centres[1] is not None:
        top_left, top_right = centres[0], centres[1]
        source_width = top_right[0] - top_left[0]
        if source_width > 0:
            source_triangle = np.float32((top_left, top_right, (top_left[0], top_left[1] + source_width)))
            target_width = PAGE_SIZE[0] - 130
            target_triangle = np.float32(((65, 65), (PAGE_SIZE[0] - 65, 65), (65, 65 + target_width)))
            matrix = cv2.getAffineTransform(source_triangle, target_triangle)
            return cv2.warpAffine(source, matrix, PAGE_SIZE, borderValue=(255, 255, 255))
    raise ValueError("Could not find the required alignment markers. Use a clear, uncropped scan with all markers visible.")


def extract_page(image_data: bytes, page_number: int) -> dict[str, list[Image.Image]]:
    """Perspective-correct one filled sheet and crop its predetermined glyph cells."""
    pages = sheet_cells()
    if page_number < 1 or page_number > len(pages):
        raise ValueError("Invalid sheet number.")
    image = Image.open(io.BytesIO(image_data))
    aligned = _align(image)
    result: dict[str, list[Image.Image]] = {}
    for cell in pages[page_number - 1]:
        x1, y1, x2, y2 = cell.bounds
        crop = aligned[y1 + 55 : y2 - 25, x1 + 25 : x2 - 25]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ink = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)[1]
        points = cv2.findNonZero(ink)
        if points is None:
            continue
        x, y, width, height = cv2.boundingRect(points)
        if width < 4 or height < 4:
            continue
        padding = 5
        glyph = gray[max(0, y - padding) : min(gray.shape[0], y + height + padding), max(0, x - padding) : min(gray.shape[1], x + width + padding)]
        result.setdefault(cell.character, []).append(Image.fromarray(glyph).convert("L"))
    return result


def extract_pdf_document(document_data: bytes) -> dict[str, list[Image.Image]]:
    """Extract all required sheet pages from a single scanned PDF."""
    document = pdfium.PdfDocument(document_data)
    expected_pages = len(sheet_cells())
    if len(document) != expected_pages:
        raise ValueError(f"The handwriting PDF must contain {expected_pages} pages; received {len(document)}.")
    glyphs: dict[str, list[Image.Image]] = {}
    for page_number, page in enumerate(document, start=1):
        rendered = page.render(scale=2).to_pil()
        buffer = io.BytesIO()
        rendered.save(buffer, format="PNG")
        for character, images in extract_page(buffer.getvalue(), page_number).items():
            glyphs.setdefault(character, []).extend(images)
    return glyphs


def save_glyphs(glyphs: dict[str, list[Image.Image]], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for character, images in glyphs.items():
        for index, image in enumerate(images):
            image.save(destination / f"{ord(character):03}_{index}.png")


def load_glyphs(source: Path) -> dict[str, list[Image.Image]]:
    glyphs: dict[str, list[Image.Image]] = {}
    for path in source.glob("*.png"):
        character = chr(int(path.stem.split("_", 1)[0]))
        glyphs.setdefault(character, []).append(Image.open(path).convert("L"))
    return glyphs
