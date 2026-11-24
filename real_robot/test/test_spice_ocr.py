"""Offline tests for helpers/spice_ocr.py: no ROS, hardware, or GUI (needs models/crnn_en.onnx,
downloaded on first use)."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helpers.spice_ocr import SpiceReader, ctc_log_likelihood, order_quad  # noqa: E402

DATA = Path(__file__).resolve().parent / 'data'


class SpiceOcrTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = SpiceReader()

    def test_real_handwritten_salt_upside_down(self):
        image = cv2.imread(str(DATA / 'salt_label_far.png'))
        labels = self.reader.read(image)
        found = [l for l in labels if l.name]
        self.assertEqual([l.name for l in found], ['salt'])        # bowl / tags / skin are not read as spices
        salt = found[0]
        self.assertGreater(salt.margin, 2.0)
        self.assertLess(np.hypot(salt.center[0] - 648, salt.center[1] - 368), 10)
        self.assertGreater(abs(salt.angle_deg), 150)                # the label is upside down in this view

    def test_real_printed_salt_next_to_shiny_lid(self):
        # printed Comic label, ~10 px tall word, touching the lid's highlight (merges with the paper blob)
        image = cv2.imread(str(DATA / 'salt_printed_far.png'))
        found = [l for l in self.reader.read(image) if l.name]
        self.assertEqual([l.name for l in found], ['salt'])
        self.assertLess(np.hypot(found[0].center[0] - 491, found[0].center[1] - 273), 10)

    def test_roi_limits_search(self):
        image = cv2.imread(str(DATA / 'salt_label_far.png'))
        self.assertIsNotNone(self.reader.find(image, 'salt', roi=(560, 300, 180, 150)))
        self.assertIsNone(self.reader.find(image, 'salt', roi=(0, 0, 300, 250)))

    def test_printed_word_any_rotation(self):
        paper = np.full((80, 220, 3), (40, 90, 160), np.uint8)    # wood-coloured background
        cv2.rectangle(paper, (20, 10), (200, 70), (245, 245, 245), -1)
        cv2.putText(paper, 'pepper', (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2, cv2.LINE_AA)
        for k in range(4):
            image = cv2.copyMakeBorder(np.ascontiguousarray(np.rot90(paper, k)), 40, 40, 40, 40,
                                       cv2.BORDER_CONSTANT, value=(40, 90, 160))
            labels = self.reader.read(image)
            self.assertEqual(labels[0].name, 'pepper', f'rotation {k * 90}')

    def test_ctc_prefers_true_word(self):
        # frames spelling "salt" with blanks; repeated letters (as in "pepper") need a blank between
        charset = '0123456789abcdefghijklmnopqrstuvwxyz'
        path = [0, charset.index('s') + 1, 0, charset.index('a') + 1, charset.index('l') + 1, 0,
                charset.index('t') + 1, 0]
        logp = np.full((len(path), 37), np.log(0.01 / 36))
        logp[np.arange(len(path)), path] = np.log(0.99)
        self.assertGreater(ctc_log_likelihood(logp, 'salt'), ctc_log_likelihood(logp, 'sat') + 3)

    def test_order_quad(self):
        q = order_quad([[10, 0], [10, 5], [0, 5], [0, 0]])
        np.testing.assert_array_equal(q, [[0, 0], [10, 0], [10, 5], [0, 5]])


if __name__ == '__main__':
    unittest.main()
