"""Which lid is which spice: the paper labels read by helpers/spice_ocr.py, matched to the Hough lids.

Each confidently read label (name not None) is put on the lid-top plane (its image centre's ray cut
with the horizontal plane at that lid's height) and given to the nearest lid within `max_dist` of
the lid's centre. The label sits on the lid or on the jar right next to it, so a few cm is enough;
a label further away than that (a stray paper on the table) belongs to no lid.

Choosing the target jar (choose_lid):
  * index mode  (--index N): lid N in reading order, the labels are only shown;
  * label mode  (default):   the one lid whose label reads the wanted spice. One lid in view and no
                             label read on it: that lid (with a warning: nothing says it is wrong).
"""
import math

import cv2
import numpy as np

_reader = None


def get_reader(cfg):
    """The CRNN reader, made once (it loads a 33 MB model; downloaded on first use)."""
    global _reader
    if _reader is None:
        from helpers.spice_ocr import SpiceReader
        _reader = SpiceReader(cfg.get('spices'), min_score=cfg.get('min_score', -2.0),
                              min_margin=cfg.get('min_margin', 1.0))
    return _reader


def pixel_on_plane(pixel, z, T_base_cam, K, D):
    """Image pixel -> (x, y, z) on the horizontal plane at height z in base_link, or None."""
    px = np.asarray(pixel, np.float64).reshape(1, 1, 2)
    ray = cv2.undistortPoints(px, np.asarray(K, float), None if D is None else np.asarray(D, float)).reshape(2)
    d = T_base_cam[:3, :3] @ np.array([ray[0], ray[1], 1.0])
    o = T_base_cam[:3, 3]
    if d[2] > -1e-6:                                   # ray does not go down to the plane
        return None
    return o + (z - o[2]) / d[2] * d


def assign_labels(lids, labels, T_base_cam, K, D, max_dist):
    """Set lid.label / lid.label_score from the read labels. -> list of (label, lid or None)."""
    for lid in lids:
        lid.label, lid.label_score = None, None
    pairs = []
    for label in labels:
        if label.name is None:
            pairs.append((label, None))
            continue
        best, best_d = None, max_dist
        for lid in lids:
            p = pixel_on_plane(label.center, lid.center[2], T_base_cam, K, D)
            if p is None:
                continue
            dist = math.dist(p[:2], lid.center[:2])
            if dist <= best_d:
                best, best_d = lid, dist
        if best is not None and (best.label is None or label.score > best.label_score):
            best.label, best.label_score = label.name, label.score
        pairs.append((label, best))
    return pairs


def holes_direction(label, lid_centre, T_base_cam, K, D):
    """Where the shaker's holes are, from its label seen from above: on our jars the holes are on
    the side of the lid "below" the text (the label's top-left -> bottom-left direction).
    -> (unit (x, y) in base_link from the lid centre towards the holes, check) where check is the
    angle (deg) between that and "opposite the label" (the label sits across the lid from the
    holes), or (None, None) if the label cannot be put on the lid plane."""
    z = float(lid_centre[2])
    q = np.asarray(label.quad, float)                 # reading order: TL, TR, BR, BL
    down_px = (q[3] - q[0]) + (q[2] - q[1])           # left and right edges, top -> bottom
    centre = np.asarray(label.center, float)
    a = pixel_on_plane(centre, z, T_base_cam, K, D)
    b = pixel_on_plane(centre + 20.0 * down_px / max(np.linalg.norm(down_px), 1e-9), z, T_base_cam, K, D)
    if a is None or b is None:
        return None, None
    d = (b - a)[:2]
    if np.linalg.norm(d) < 1e-9:
        return None, None
    d = d / np.linalg.norm(d)
    away = np.asarray(lid_centre, float)[:2] - a[:2]  # from the label across the lid
    check = None
    if np.linalg.norm(away) > 0.003:                  # label far enough off centre to say
        check = math.degrees(math.acos(float(np.clip(d @ away / np.linalg.norm(away), -1.0, 1.0))))
    return d, check


def choose_lid(lids, spice=None, index=None):
    """-> (lid, note) or raises ValueError with what to do. `lids` in reading order (lid k = lids[k])."""
    if not lids:
        raise ValueError('no lids')
    if index is not None:
        if not 0 <= index < len(lids):
            raise ValueError(f'--index {index}: only {len(lids)} lid(s) seen (0..{len(lids) - 1})')
        lid = lids[index]
        return lid, f'lid {index} (--index)' + (f', label reads "{lid.label}"' if lid.label else '')
    read = ', '.join(f'lid {k}: "{lid.label}"' for k, lid in enumerate(lids) if lid.label) or 'none read'
    if spice is None:
        if len(lids) == 1:
            return lids[0], 'the only lid'
        raise ValueError(f'{len(lids)} lids seen and no spice to look for: give --spice NAME or --index N')
    matches = [k for k, lid in enumerate(lids) if lid.label == spice]
    if len(matches) == 1:
        return lids[matches[0]], f'lid {matches[0]}, label reads "{spice}"'
    if len(matches) > 1:
        raise ValueError(f'{len(matches)} lids read "{spice}" ({", ".join(map(str, matches))}): '
                         f'pick one with --index N')
    if len(lids) == 1 and lids[0].label is None:
        return lids[0], f'the only lid; WARNING: its label was not read, "{spice}" not confirmed'
    raise ValueError(f'no lid with a "{spice}" label among {len(lids)} lid(s) (labels: {read}); '
                     f'pick one with --index N')


def draw_labels(image, pairs, lids):
    """Outlines of the labels read as a spice: green = on a lid (its name is in the lid's caption),
    orange = on no lid (with its name). Text patches that read as no spice are not drawn."""
    for label, lid in pairs:
        if label.name is None:
            continue
        colour = (0, 200, 0) if lid is not None else (0, 140, 255)
        cv2.polylines(image, [label.quad.astype(np.int32)], True, colour, 2)
        if lid is None:
            org = (int(label.quad[:, 0].min()), int(label.quad[:, 1].max()) + 18)
            text = f'{label.name}: on no lid'
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
    return image


def lid_caption(k, lid, target=False):
    """Short caption for lid k, drawn under its circle: 'lid 4 salt' / 'z +0.025' (/ 'TARGET')."""
    text = f'lid {k}' + (f' {lid.label}' if lid.label else '') + f'\nz {lid.center[2]:+.3f}'
    return text + ('\nTARGET' if target else '')
