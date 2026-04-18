import cv2
import numpy as np

# lazy load — easyocr import is slow and pulls in a lot of stuff
_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader


def _conf_color(conf):
    """Red → green gradient based on confidence."""
    return (0, int(255 * conf), int(255 * (1 - conf)))


def detect_text(image):
    """OCR the image with EasyOCR. Returns (annotated_image, detections)."""
    reader = _get_reader()
    annotated = image.copy()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb, paragraph=False)

    detections = []

    for (bbox, text, conf) in results:
        if conf < 0.1:
            continue

        pts = np.array(bbox, dtype=np.int32)
        x1, y1 = pts[0]
        x2, y2 = pts[2]
        color = _conf_color(conf)

        cv2.polylines(annotated, [pts], True, color, 2)

        label = f'"{text}" {conf:.0%}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ly = max(int(y1) - 2, th + 6)
        cv2.rectangle(annotated, (int(x1), ly - th - 6), (int(x1) + tw + 8, ly + 2), color, -1)
        cv2.putText(annotated, label, (int(x1) + 4, ly - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        detections.append({
            "Text": text, "Confidence": f"{conf:.2%}",
            "X1": int(x1), "Y1": int(y1), "X2": int(x2), "Y2": int(y2),
        })

    return annotated, detections
