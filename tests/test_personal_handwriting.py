import unittest

import cv2
import numpy as np
from PIL import Image

from app.core.personal_handwriting import PAGE_SIZE, create_sheets, extract_page, extract_pdf_document, sheet_cells


class PersonalHandwritingTests(unittest.TestCase):
    def test_templates_cover_each_character_three_times(self) -> None:
        cells = [cell for page in sheet_cells() for cell in page]

        self.assertEqual(len(create_sheets()), len(sheet_cells()))
        self.assertEqual(len(cells), 3 * len({cell.character for cell in cells}))

    def test_extract_page_crops_known_cell(self) -> None:
        page = create_sheets()[0]
        cell = sheet_cells()[0][0]
        image = np.array(page)
        x1, y1, x2, y2 = cell.bounds
        cv2.putText(image, cell.character, (x1 + 100, y1 + 210), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8)
        encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()

        glyphs = extract_page(encoded, 1)

        self.assertIn(cell.character, glyphs)
        self.assertGreater(glyphs[cell.character][0].size[0], 4)
        self.assertEqual(PAGE_SIZE, page.size)

    def test_pdf_document_requires_all_sheet_pages(self) -> None:
        document = create_sheets()[0]
        pdf = __import__("io").BytesIO()
        document.save(pdf, format="PDF")

        with self.assertRaisesRegex(ValueError, "must contain"):
            extract_pdf_document(pdf.getvalue())
