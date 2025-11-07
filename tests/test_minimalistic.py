import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image

from src.ascii_art_generator import minimalistic


class TestMinimalistic(unittest.TestCase):

    def setUp(self):
        self.test_image = Image.new("RGB", (10, 10), "white")

    def test_create_background_mask(self):
        mask = minimalistic.create_background_mask(self.test_image)
        self.assertEqual(mask.mode, "L")



if __name__ == "__main__":
    unittest.main()
