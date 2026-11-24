"""Small OpenCV drawing helpers shared by the live tools."""
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_outlined_text(image, text, origin, scale, color, thickness=2, outline=None):
    """Text with a black outline, readable on any background. The outline is drawn `outline` px
    thick (default thickness + 3) underneath the coloured text."""
    origin = (int(origin[0]), int(origin[1]))
    cv2.putText(image, text, origin, FONT, scale, (0, 0, 0), thickness + 3 if outline is None else outline)
    cv2.putText(image, text, origin, FONT, scale, color, thickness)
