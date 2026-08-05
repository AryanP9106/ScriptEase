"""Render typed text with randomized glyphs from EMNIST ByClass."""

from __future__ import annotations

import gzip
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


EMNIST_CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
IMAGE_FILE = "emnist-byclass-train-images-idx3-ubyte.gz"
LABEL_FILE = "emnist-byclass-train-labels-idx1-ubyte.gz"


@dataclass(frozen=True)
class RenderOptions:
    font_size: int = 30
    line_spacing: int = 18
    margin: int = 100
    ink_color: tuple[int, int, int] = (20, 39, 70)
    seed: int | None = None


class HandwritingLibrary:
    """A small randomized pool of handwritten EMNIST glyphs per character."""

    def __init__(self, glyphs: dict[str, list[Image.Image]]) -> None:
        self.glyphs = glyphs

    @classmethod
    def from_emnist(cls, dataset_dir: Path, samples_per_character: int = 24) -> "HandwritingLibrary":
        """Load a bounded, representative glyph set without loading all EMNIST images."""
        images_path = dataset_dir / IMAGE_FILE
        labels_path = dataset_dir / LABEL_FILE
        if not images_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(
                "EMNIST ByClass training files were not found. Download and extract the dataset first."
            )

        rng = random.Random(0)
        selected: dict[str, list[Image.Image]] = defaultdict(list)
        seen: dict[str, int] = defaultdict(int)

        with gzip.open(images_path, "rb") as image_stream, gzip.open(labels_path, "rb") as label_stream:
            image_stream.read(16)
            label_stream.read(8)
            while True:
                label = label_stream.read(1)
                pixels = image_stream.read(784)
                if not label or len(pixels) != 784:
                    break

                character = EMNIST_CHARACTERS[label[0]]
                seen[character] += 1
                image = Image.fromarray(np.frombuffer(pixels, dtype=np.uint8).reshape(28, 28).T).convert("L")
                glyphs = selected[character]
                if len(glyphs) < samples_per_character:
                    glyphs.append(image)
                elif rng.randrange(seen[character]) < samples_per_character:
                    glyphs[rng.randrange(samples_per_character)] = image

        missing = set(EMNIST_CHARACTERS) - set(selected)
        if missing:
            raise ValueError(f"EMNIST data is incomplete; missing: {''.join(sorted(missing))}")
        return cls(dict(selected))


class HandwritingRenderer:
    """Lay out a typed document on A4 pages using handwritten image glyphs."""

    page_size = (1240, 1754)

    def __init__(self, library: HandwritingLibrary, options: RenderOptions | None = None) -> None:
        self.library = library
        self.options = options or RenderOptions()
        self.random = random.Random(self.options.seed)

    def render(self, text: str) -> list[Image.Image]:
        if not text.strip():
            raise ValueError("Enter or upload some text before generating a document.")

        page = self._new_page()
        pages = [page]
        cursor_x = self.options.margin
        cursor_y = self.options.margin
        line_height = self.options.font_size + self.options.line_spacing
        max_x = self.page_size[0] - self.options.margin
        max_y = self.page_size[1] - self.options.margin

        for paragraph in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            words = paragraph.split() or [""]
            for word in words:
                word_width = self._measure_word(word)
                if cursor_x > self.options.margin and cursor_x + word_width > max_x:
                    cursor_x = self.options.margin
                    cursor_y += line_height
                if cursor_y + line_height > max_y:
                    page = self._new_page()
                    pages.append(page)
                    cursor_x = self.options.margin
                    cursor_y = self.options.margin
                cursor_x = self._draw_word(page, word, cursor_x, cursor_y)
                cursor_x += self.options.font_size // 2
            cursor_x = self.options.margin
            cursor_y += line_height
            if cursor_y + line_height > max_y:
                page = self._new_page()
                pages.append(page)
                cursor_y = self.options.margin
        return pages

    def _new_page(self) -> Image.Image:
        return Image.new("RGB", self.page_size, "white")

    def _measure_word(self, word: str) -> int:
        widths = [self._glyph_width(character) for character in word]
        return sum(widths) + max(0, len(widths) - 1) * 2

    def _glyph_width(self, character: str) -> int:
        if character in self.library.glyphs:
            glyph = self.library.glyphs[character][0]
            bbox = Image.eval(glyph, lambda value: 255 - value).getbbox()
            if bbox:
                return max(8, round((bbox[2] - bbox[0]) * self.options.font_size / 28))
        return self.options.font_size // 2

    def _draw_word(self, page: Image.Image, word: str, cursor_x: int, cursor_y: int) -> int:
        for character in word:
            width = self._draw_character(page, character, cursor_x, cursor_y)
            cursor_x += width + 2
        return cursor_x

    def _draw_character(self, page: Image.Image, character: str, cursor_x: int, cursor_y: int) -> int:
        glyphs = self.library.glyphs.get(character)
        if not glyphs:
            self._draw_fallback(page, character, cursor_x, cursor_y)
            return self.options.font_size // 2

        glyph = self.random.choice(glyphs)
        alpha = Image.eval(glyph, lambda value: 255 - value)
        bbox = alpha.getbbox()
        if not bbox:
            return self.options.font_size // 2
        alpha = alpha.crop(bbox)
        target_height = max(12, self.options.font_size + self.random.randint(-3, 3))
        target_width = max(5, round(alpha.width * target_height / alpha.height))
        alpha = alpha.resize((target_width, target_height), Image.Resampling.LANCZOS)
        alpha = alpha.rotate(self.random.uniform(-2.0, 2.0), expand=True, resample=Image.Resampling.BICUBIC)
        ink = Image.new("RGB", alpha.size, self.options.ink_color)
        x = cursor_x + self.random.randint(-1, 1)
        y = cursor_y + self.options.font_size - target_height + self.random.randint(-2, 2)
        page.paste(ink, (x, y), alpha)
        return target_width

    def _draw_fallback(self, page: Image.Image, character: str, cursor_x: int, cursor_y: int) -> None:
        draw = ImageDraw.Draw(page)
        font = ImageFont.load_default()
        draw.text((cursor_x, cursor_y + self.options.font_size // 3), character, fill=self.options.ink_color, font=font)


def save_pdf(pages: Iterable[Image.Image], destination: Path) -> None:
    """Save rendered page images as a single PDF."""
    page_list = list(pages)
    if not page_list:
        raise ValueError("Cannot save an empty document.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    page_list[0].save(destination, "PDF", save_all=True, append_images=page_list[1:], resolution=150.0)
