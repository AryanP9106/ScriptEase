import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.core.handwriting import HandwritingLibrary, HandwritingRenderer, RenderOptions, save_pdf


class HandwritingRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        glyph = Image.new("L", (28, 28), 255)
        for point in range(7, 21):
            glyph.putpixel((point, point), 0)
        self.library = HandwritingLibrary({character: [glyph] for character in "AaZz09"})

    def test_renders_multiple_pages_for_long_text(self) -> None:
        renderer = HandwritingRenderer(self.library, RenderOptions(font_size=40, margin=100, seed=1))

        pages = renderer.render(("Az09 " * 2_000).strip())

        self.assertGreater(len(pages), 1)
        self.assertEqual(pages[0].size, (1240, 1754))

    def test_saves_a_pdf(self) -> None:
        renderer = HandwritingRenderer(self.library, RenderOptions(seed=1))
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "output.pdf"

            save_pdf(renderer.render("Az09"), output_path)

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)
