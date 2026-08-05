import unittest

import cv2
import numpy as np

from preprocess import PreprocessConfig, preprocess_page, resize_long_edge


class PreprocessTests(unittest.TestCase):
    def test_resize_long_edge_preserves_aspect_ratio(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        resized = resize_long_edge(image, 100)

        self.assertEqual(resized.shape, (50, 100, 3))

    def test_pipeline_returns_binary_image(self) -> None:
        image = np.full((120, 160, 3), 255, dtype=np.uint8)
        cv2.putText(image, "Hi", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        config = PreprocessConfig(
            deskew_enabled=False,
            shadow_removal_enabled=False,
            denoise_enabled=False,
            sharpen_enabled=False,
            max_long_edge=None,
        )

        processed = preprocess_page(image, config)

        self.assertEqual(processed.ndim, 2)
        self.assertTrue(set(np.unique(processed)).issubset({0, 255}))

    def test_even_kernel_values_are_normalized(self) -> None:
        config = PreprocessConfig(shadow_blur_kernel=10, adaptive_block_size=2)

        config.validate()

        self.assertEqual(config.shadow_blur_kernel, 11)
        self.assertEqual(config.adaptive_block_size, 3)


if __name__ == "__main__":
    unittest.main()
