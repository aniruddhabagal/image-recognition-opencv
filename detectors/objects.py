import cv2
from ultralytics import YOLO

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model

# cycle through these for different classes
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0),
]


def detect_objects(image, conf=0.15):
    """Run YOLOv8 on the image. Returns (annotated_image, detections)."""
    model = _get_model()
    annotated = image.copy()

    results = model(image, conf=conf, iou=0.45, verbose=False)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        c = float(box.conf[0])
        cls = int(box.cls[0])
        name = model.names[cls]
        color = COLORS[cls % len(COLORS)]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{name} {c:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        detections.append({
            "Object": name, "Confidence": f"{c:.2%}",
            "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
        })

    return annotated, detections
