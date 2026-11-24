"""Is the jar in a pocket? Compare a picture of the pocket now with reference pictures of it taken
earlier ("empty" and/or "with the jar"). No ROS in here.

Each picture is turned into a top-down patch of the pocket: the colour image warped onto the
horizontal plane at the pocket's rim height, a square around the pocket centre, 1 mm per pixel.
In that view the pocket is in the same place whatever the exact camera pose was, so two patches
can be compared pixel by pixel. The difference is the fraction of the pocket's disc whose grey
level changed by more than `grey_step` (after matching the brightness of the patch corners,
outside the pocket, so a slightly different exposure does not count).

  * empty and full reference: the jar is there if the patch is more like the full one;
  * one reference only: there must be (empty) / must not be (full) a big change.
"""
import cv2
import numpy as np

from spice_task.lid_detector import warp_plane


def patch(image, T_base_cam, K, D, centre, radius, res=0.001):
    """Top-down grey patch of the plane at centre[2], 2 * radius wide, centred on centre[:2]."""
    x_range = (centre[0] - radius, centre[0] + radius)
    y_range = (centre[1] - radius, centre[1] + radius)
    bev = warp_plane(image, T_base_cam, K, D, centre[2], x_range, y_range, res)
    return cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY) if bev.ndim == 3 else bev


def difference(a, b, grey_step=30, blur=5):
    """Fraction of the inscribed disc where patches a and b differ (0 = same, 1 = all changed)."""
    n = min(a.shape[0], b.shape[0], a.shape[1], b.shape[1])
    a, b = a[:n, :n], b[:n, :n]
    disc = np.zeros((n, n), np.uint8)
    cv2.circle(disc, (n // 2, n // 2), n // 2 - 1, 1, -1)
    both = (a > 0) & (b > 0)                         # warp leaves 0 outside the image
    seen = both & (disc > 0)
    if np.count_nonzero(seen) < 0.5 * np.count_nonzero(disc):
        return float('nan')
    k = blur | 1
    fa = cv2.GaussianBlur(a, (k, k), 0).astype(np.float32)
    fb = cv2.GaussianBlur(b, (k, k), 0).astype(np.float32)
    # exposure: from the corners outside the disc (the rack / stand around the pocket), which the
    # jar does not change -- not from the disc itself, or a uniformly brighter lid would cancel out
    around = both & (disc == 0)
    if np.count_nonzero(around) > 50:
        fb += np.median(fa[around]) - np.median(fb[around])
    changed = (np.abs(fa - fb) > grey_step) & seen
    return float(np.count_nonzero(changed)) / float(np.count_nonzero(seen))


def judge(now, empty=None, full=None, changed=0.25, grey_step=30):
    """(jar there?, message). None for 'cannot tell' (no reference, or the pocket not in view)."""
    d_empty = difference(now, empty, grey_step) if empty is not None else None
    d_full = difference(now, full, grey_step) if full is not None else None
    vals = [d for d in (d_empty, d_full) if d is not None]
    if not vals or any(np.isnan(d) for d in vals):
        return None, 'pocket not (fully) in view or no reference picture'
    parts = []
    if d_empty is not None:
        parts.append(f'{100 * d_empty:.0f}% changed from the empty picture')
    if d_full is not None:
        parts.append(f'{100 * d_full:.0f}% changed from the with-jar picture')
    msg = ', '.join(parts)
    if d_empty is not None and d_full is not None:
        return d_full < d_empty, msg
    if d_empty is not None:
        return d_empty > changed, msg
    return d_full < changed, msg


def side_by_side(patches, labels, scale=4):
    """Patches next to each other, enlarged and labelled, for the captures folder / RViz."""
    tiles = []
    for p, label in zip(patches, labels):
        if p is None:
            continue
        t = cv2.resize(p, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        h = t.shape[0]
        n = t.shape[0] // 2
        cv2.circle(t, (n, n), n - 2, (255, 255, 0), 1)
        cv2.putText(t, label, (6, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(t, label, (6, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(t)
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for t in tiles]
    return np.hstack(tiles)
