import cv2
import numpy as np


def detect_edges(image, min_area=300):
    """Canny edge detection + contour extraction. Returns (annotated_image, detections)."""
    annotated = image.copy()
    img_h, img_w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # auto-pick canny thresholds from the median intensity
    med = np.median(blurred)
    lo = int(max(0, 0.5 * med))
    hi = int(min(255, 1.5 * med))
    edges = cv2.Canny(blurred, lo, hi)

    # slight dilation to connect broken edges
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kern, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # random but deterministic color palette
    np.random.seed(42)
    palette = [(int(c[0]), int(c[1]), int(c[2]))
               for c in np.random.randint(80, 255, (200, 3))]

    detections = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w > img_w * 0.95 and h > img_h * 0.95:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        verts = len(approx)

        if verts == 3:
            shape = "Triangle"
        elif verts == 4:
            shape = "Rectangle"
        elif verts == 5:
            shape = "Pentagon"
        elif verts > 8:
            shape = "Circle/Ellipse"
        else:
            shape = f"Polygon ({verts} pts)"

        color = palette[i % len(palette)]
        cv2.drawContours(annotated, [cnt], -1, color, 2)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 1)

        label = f"{shape} A={area:.0f}"
        (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        ly = max(y - 2, th_t + 6)
        cv2.rectangle(annotated, (x, ly - th_t - 4), (x + tw + 6, ly + 2), color, -1)
        cv2.putText(annotated, label, (x + 3, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

        detections.append({
            "Shape": shape, "Area": int(area), "Vertices": verts,
            "X": x, "Y": y, "Width": w, "Height": h,
        })

    detections.sort(key=lambda d: d["Area"], reverse=True)
    return annotated, detections
