"""Identify spice shakers from a handwritten/printed paper label (replaces AprilTag IDs).

Pipeline, per color image (optionally only inside an ROI, e.g. the shaker box from the CV localizer):
  1. find     two candidate finders, both always run:
              paper  bright, low-saturation, roughly rectangular blobs (works for thin handwriting)
              text   dark strokes surrounded by bright pixels, grouped into words (works when the
                     label touches a shiny lid, which merges with the paper blob)
  2. rectify  warp each candidate's rectangle to an upright crop and cut out the inked word;
              overlapping candidates keep the best reading
  3. read     CRNN text recognizer (OpenCV dnn, no extra packages). Instead of trusting the free
              read-out, every name in the vocabulary is scored by its CTC likelihood, in all 4
              rotations (the label can face any way). This is far more robust on small, shaky
              handwriting than fuzzy-matching a misread string.
  4. accept   best name if its per-character log-likelihood >= min_score AND it beats the runner-up
              by >= min_margin; otherwise the label is reported as unknown (name=None).

Usage:
    reader = SpiceReader(['salt', 'pepper'])
    for label in reader.read(bgr_image):             # or reader.read(bgr_image, roi=(x, y, w, h))
        print(label.name, label.score, label.center)

Model: OpenCV Zoo text_recognition_CRNN_EN_2021sep.onnx (charset 0-9a-z), downloaded to
models/crnn_en.onnx on first use.
"""
import os
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, 'models', 'crnn_en.onnx')
MODEL_URL = ('https://github.com/opencv/opencv_zoo/raw/main/models/text_recognition_crnn/'
             'text_recognition_CRNN_EN_2021sep.onnx')
CHARSET = '0123456789abcdefghijklmnopqrstuvwxyz'      # CTC index 0 = blank, i + 1 = CHARSET[i]
DEFAULT_SPICES = ['salt', 'pepper', 'paprika', 'cumin', 'oregano', 'garlic', 'sugar', 'cinnamon',
                  'basil', 'thyme', 'chili', 'turmeric', 'ginger', 'nutmeg', 'parsley', 'rosemary']


@dataclass
class Label:
    name: str | None          # vocabulary word, or None if not confident
    score: float              # per-character CTC log-likelihood of the best word (0 = perfect)
    margin: float             # best score - runner-up score
    raw: str                  # free (greedy) read-out, for debugging
    quad: np.ndarray          # (4, 2) paper (or word) corners in image pixels, reading order: TL, TR, BR, BL
    center: tuple             # (u, v) pixels
    angle_deg: float          # reading direction (TL -> TR) in the image, deg, atan2(dv, du)
    source: str = 'paper'     # which finder produced it: 'paper' or 'text'
    scores: dict = field(default_factory=dict, repr=False)


def ensure_model(path=MODEL_PATH):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f'[spice_ocr] downloading CRNN model to {path} ...')
        urllib.request.urlretrieve(MODEL_URL, path + '.part')
        os.replace(path + '.part', path)
    return path


def ctc_log_likelihood(logp, word):
    """log P(word | per-frame log-probs (T, C)) under CTC, blank = 0."""
    labels = [0]
    for c in word:
        labels += [CHARSET.index(c) + 1, 0]
    labels = np.array(labels)
    S = len(labels)
    # transitions from s-2 allowed when labels[s] is a character different from labels[s-2]
    skip = np.zeros(S, bool)
    skip[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    alpha = np.full(S, -np.inf)
    alpha[0] = logp[0, labels[0]]
    alpha[1] = logp[0, labels[1]]
    for t in range(1, len(logp)):
        prev1 = np.concatenate(([-np.inf], alpha[:-1]))
        prev2 = np.where(skip, np.concatenate(([-np.inf, -np.inf], alpha[:-2])), -np.inf)
        alpha = np.logaddexp(np.logaddexp(alpha, prev1), prev2) + logp[t, labels]
    return float(np.logaddexp(alpha[-1], alpha[-2]))


def order_quad(box):
    """4 corners -> TL, TR, BR, BL for an image with y down (by angle around the centre)."""
    box = np.asarray(box, np.float32)
    c = box.mean(axis=0)
    ang = np.arctan2(box[:, 1] - c[1], box[:, 0] - c[0])
    box = box[np.argsort(ang)]                         # starts near -pi: top-left-ish, clockwise on screen
    start = np.argmin(box.sum(axis=1))                 # smallest u+v = top-left
    return np.roll(box, -start, axis=0)


class SpiceReader:
    def __init__(self, spices=None, model_path=None, min_score=-2.0, min_margin=1.0,
                 min_area=300, max_area=40000, max_saturation=70, min_value=150,
                 word_max_saturation=120, word_min_value=170, word_contrast=0.7):
        self.spices = [s.lower() for s in (spices or DEFAULT_SPICES)]
        bad = [s for s in self.spices if not s or any(c not in CHARSET for c in s)]
        if bad:
            raise ValueError(f'spice names must be lowercase a-z/0-9 only: {bad}')
        self.net = cv2.dnn.readNet(ensure_model(model_path or MODEL_PATH))
        self.min_score, self.min_margin = min_score, min_margin
        self.min_area, self.max_area = min_area, max_area
        self.max_saturation, self.min_value = max_saturation, min_value
        # text finder: "bright" around ink is looser than for paper blobs (tinted paper under the camera's
        # white balance); ink must be darker than word_contrast x its surroundings (edges/shadows aren't)
        self.word_max_saturation, self.word_min_value = word_max_saturation, word_min_value
        self.word_contrast = word_contrast

    # ------------------------------------------------------------------ 1. find paper labels
    def find_labels(self, image, roi=None):
        """-> list of (4, 2) quads (min-area rectangles of white paper), largest first."""
        x0, y0 = 0, 0
        if roi is not None:
            x, y, w, h = (int(round(v)) for v in roi)
            x0, y0 = max(0, x), max(0, y)
            image = image[y0:y + h, x0:x + w]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = ((hsv[..., 1] < self.max_saturation) & (hsv[..., 2] > self.min_value)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))   # ink strokes -> paper
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quads = []
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if not self.min_area <= area <= self.max_area:
                continue
            (cx, cy), (w, h), _ = rect = cv2.minAreaRect(c)
            if min(w, h) < 12 or area / (w * h) < 0.45 or max(w, h) / min(w, h) > 5:
                continue
            quads.append(cv2.boxPoints(rect) + np.float32([x0, y0]))
        return quads

    def find_words(self, image, roi=None):
        """Text-first candidates: dark ink strokes surrounded by bright pixels, grouped into words.
        -> list of (4, 2) word quads (padded a little). Independent of the paper's outline, so a
        label lying against a shiny lid (which looks like paper too) is still cut out cleanly."""
        x0, y0 = 0, 0
        if roi is not None:
            x, y, w, h = (int(round(v)) for v in roi)
            x0, y0 = max(0, x), max(0, y)
            image = image[y0:y + h, x0:x + w]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        bright = (hsv[..., 1] < self.word_max_saturation) & (hsv[..., 2] > self.word_min_value)
        # black-hat = how much darker a pixel is than its neighbourhood: thin dark strokes light up
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        n, lab, stats, _ = cv2.connectedComponentsWithStats((blackhat > 35).astype(np.uint8), connectivity=8)
        H, W = gray.shape
        keep = np.zeros(gray.shape, np.uint8)
        heights = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 3 or area > 3000 or max(w, h) > 120:
                continue
            ax0, ay0, ax1, ay1 = max(0, x - 3), max(0, y - 3), min(W, x + w + 3), min(H, y + h + 3)
            comp = (lab[ay0:ay1, ax0:ax1] == i).astype(np.uint8)
            ring = (cv2.dilate(comp, np.ones((5, 5), np.uint8)) - comp) > 0
            if not ring.any() or bright[ay0:ay1, ax0:ax1][ring].mean() < 0.6:    # not on paper
                continue
            g = gray[ay0:ay1, ax0:ax1].astype(np.float32)
            if g[comp > 0].mean() > self.word_contrast * g[ring].mean():         # shadow / edge, not ink
                continue
            keep[lab == i] = 1
            heights.append(h)
        if not heights:
            return []
        k = int(np.clip(np.median(heights) * 0.7, 3, 25))                         # joins letters of a word
        groups = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        m, glab = cv2.connectedComponents(groups, connectivity=8)
        quads = []
        for j in range(1, m):
            pts = np.column_stack(np.nonzero((glab == j) & (keep > 0)))[:, ::-1].astype(np.float32)
            if len(pts) < 8:
                continue
            (cx, cy), (w, h), a = cv2.minAreaRect(pts)
            if w < h:
                w, h, a = h, w, a + 90
            if w < 8 or h < 4 or w > 300:
                continue
            box = cv2.boxPoints(((cx, cy), (w + 0.5 * h + 4, 1.5 * h + 4), a))
            quads.append(box + np.float32([x0, y0]))
        return quads

    # ------------------------------------------------------------------ 2. rectify + cut out the word
    @staticmethod
    def rectify(image, quad):
        """Warp the paper to an upright crop, long side horizontal. -> (gray crop, quad in crop order)."""
        q = order_quad(quad)
        w = int(round(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))))
        h = int(round(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))))
        if h > w:                                     # make the long side horizontal
            q = np.roll(q, -1, axis=0)
            w, h = h, w
        dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(q, dst), (w, h))
        return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), q

    @staticmethod
    def word_box(gray):
        """Bounding box of the ink on the paper, ignoring the paper's edges and whatever else is in the
        crop (the paper blob can include a shiny lid). -> (x0, y0, x1, y1) or None."""
        h, w = gray.shape
        # the paper = the largest bright region (Otsu), holes (the ink) filled, shrunk off its edges
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        # paper is the brightest thing in the crop; a shiny lid can be close to it, so also stay near the top
        bright = (blur > max(otsu, np.percentile(blur, 95) - 35)).astype(np.uint8) * 255
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))    # cut thin bridges
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))   # fill the ink
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        paper = np.zeros_like(gray)
        cv2.drawContours(paper, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        paper = cv2.erode(paper, np.ones((5, 5), np.uint8))
        ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                    15 if min(h, w) > 20 else 7, 12)
        inner = cv2.bitwise_and(ink, paper)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(inner, connectivity=8)
        # drop specks and long thin horizontal lines (paper edges / shadows), then keep the strokes near the
        # main ones (the letters), so a stray mark in a corner doesn't stretch the box
        comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 3
                 and stats[i, cv2.CC_STAT_WIDTH] <= 6 * max(1, stats[i, cv2.CC_STAT_HEIGHT])]
        if not comps:
            return None
        big = max(stats[i, cv2.CC_STAT_AREA] for i in comps)
        main = [i for i in comps if stats[i, cv2.CC_STAT_AREA] >= 0.2 * big]
        mx0 = min(stats[i, 0] for i in main) - 0.5 * h
        mx1 = max(stats[i, 0] + stats[i, 2] for i in main) + 0.5 * h
        my0 = min(stats[i, 1] for i in main) - 0.15 * h
        my1 = max(stats[i, 1] + stats[i, 3] for i in main) + 0.15 * h
        keep = [i for i in comps if stats[i, 0] >= mx0 and stats[i, 0] + stats[i, 2] <= mx1
                and stats[i, 1] >= my0 and stats[i, 1] + stats[i, 3] <= my1]
        xs0 = min(stats[i, 0] for i in keep)
        ys0 = min(stats[i, 1] for i in keep)
        xs1 = max(stats[i, 0] + stats[i, 2] for i in keep)
        ys1 = max(stats[i, 1] + stats[i, 3] for i in keep)
        pad_y, pad_x = 4, 6
        return max(0, xs0 - pad_x), max(0, ys0 - pad_y), min(w, xs1 + pad_x), min(h, ys1 + pad_y)

    # ------------------------------------------------------------------ 3. recognize
    def log_probs(self, gray):
        blob = cv2.dnn.blobFromImage(cv2.resize(gray, (100, 32), interpolation=cv2.INTER_CUBIC),
                                     1 / 127.5, (100, 32), 127.5)
        self.net.setInput(blob)
        out = self.net.forward()
        out = out.reshape(out.shape[0], -1).astype(np.float64)   # (T, 37)
        out -= out.max(axis=1, keepdims=True)
        return out - np.log(np.exp(out).sum(axis=1, keepdims=True))

    @staticmethod
    def greedy(logp):
        best = logp.argmax(axis=1)
        return ''.join(CHARSET[i - 1] for j, i in enumerate(best) if i and (j == 0 or i != best[j - 1]))

    def read_word(self, gray, rotations=(0, 1, 2, 3)):
        """Score every spice in the given quarter-turn rotations of a word crop.
        -> (name, score, margin, raw, rotation k, scores)."""
        best = {}                                     # spice -> (score, k, raw)
        raws = {}
        for k in rotations:
            g = np.ascontiguousarray(np.rot90(gray, k))
            if k % 2:                                 # rotated by 90: re-crop so the word fills it
                box = self.word_box(g)
                if box is not None:
                    g = g[box[1]:box[3], box[0]:box[2]]
            if g.shape[0] < 6 or g.shape[1] < 6:
                continue
            logp = self.log_probs(g)
            raws[k] = self.greedy(logp)
            for s in self.spices:
                sc = ctc_log_likelihood(logp, s) / len(s)
                if s not in best or sc > best[s][0]:
                    best[s] = (sc, k, raws[k])
        if not best:
            return None, -np.inf, 0.0, '', 0, {}
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
        name, (score, k, raw) = ranked[0]
        margin = score - ranked[1][1][0] if len(ranked) > 1 else np.inf
        return name, score, margin, raw, k, {s: v[0] for s, v in best.items()}

    # ------------------------------------------------------------------ all together
    def read(self, image, roi=None):
        """All labels in a BGR image (or inside roi=(x, y, w, h)). -> list[Label], confident ones first."""
        candidates = [(q, 'paper') for q in self.find_labels(image, roi)]
        candidates += [(q, 'text') for q in self.find_words(image, roi)]
        labels = []
        for quad, source in candidates:
            gray, q = self.rectify(image, quad)
            if source == 'paper':
                box = self.word_box(gray)
                if box is None:
                    continue
                word = gray[box[1]:box[3], box[0]:box[2]]
                # report the word's outline, not the blob's: a shiny lid can merge with the paper into
                # one big bright blob (the box then jumped between the lid and the label frame to frame)
                h, w = gray.shape
                to_crop = cv2.getPerspectiveTransform(
                    q.astype(np.float32), np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]))
                corners = np.float32([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
                q = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), np.linalg.inv(to_crop)).reshape(4, 2)
            else:
                word = gray                                # already cut to the word
            # a clearly horizontal word only needs 0 / 180 deg; near-square crops get all four
            rotations = (0, 2) if word.shape[1] > 1.8 * word.shape[0] else (0, 1, 2, 3)
            name, score, margin, raw, k, scores = self.read_word(word, rotations)
            # word read after k CCW quarter turns of the crop -> the reading-order corners are q rolled by k
            q = np.roll(q, -k, axis=0)
            accepted = score >= self.min_score and margin >= self.min_margin
            du, dv = q[1] - q[0]
            labels.append(Label(name if accepted else None, score, margin, raw, q,
                                 tuple(q.mean(axis=0).tolist()), float(np.degrees(np.arctan2(dv, du))),
                                 source, scores))
        # overlapping candidates (paper blob + word in it) describe one label: keep the best reading
        labels.sort(key=lambda l: (l.name is None, -l.score))
        kept = []
        for l in labels:
            if all(cv2.pointPolygonTest(o.quad.astype(np.float32), l.center, False) < 0 and
                   cv2.pointPolygonTest(l.quad.astype(np.float32), o.center, False) < 0 for o in kept):
                kept.append(l)
        return kept

    def find(self, image, spice, roi=None):
        """The best label reading `spice`, or None."""
        return next((l for l in self.read(image, roi) if l.name == spice.lower()), None)


def draw(image, labels):
    """Draw label outlines (green = identified, red = unknown) with name and score."""
    out = image.copy()
    for l in labels:
        color = (0, 200, 0) if l.name else (0, 0, 255)
        cv2.polylines(out, [l.quad.astype(np.int32)], True, color, 2)
        cv2.circle(out, tuple(int(v) for v in l.quad[0]), 4, (0, 255, 255), -1)   # reading-order start
        text = f'{l.name or "?"} {l.score:.2f} (+{l.margin:.1f}) [{l.raw}]'
        org = (int(l.quad[:, 0].min()), max(15, int(l.quad[:, 1].min()) - 8))
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out
