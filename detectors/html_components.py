import cv2
import numpy as np

# BGR colors per component type for the overlay
COMPONENT_COLORS = {
    "Navigation Bar": (255, 165, 0),
    "Image/Card":     (0, 200, 255),
    "Button":         (50, 205, 50),
    "Input Field":    (255, 105, 180),
    "Icon":           (255, 255, 0),
    "Tab":            (147, 112, 219),
    "Container":      (100, 149, 237),
    "Link/Text Block":(200, 200, 200),
}


def classify_component(x, y, w, h, img_w, img_h):
    """Figure out what kind of UI component a bounding rect most likely is."""
    aspect = w / max(h, 1)
    area = w * h
    area_pct = area / (img_w * img_h)
    rel_w = w / img_w
    rel_h = h / img_h
    cy = (y + h / 2) / img_h

    # full-width thin bands → navbar
    if rel_w > 0.7 and rel_h < 0.08:
        return "Navigation Bar"

    # wide + short → search/input field
    if 3.5 < aspect < 25 and 15 < h < 60 and w > 100:
        return "Input Field"

    # small square-ish → icons
    if w < 50 and h < 50 and 0.6 < aspect < 1.7 and area > 200:
        return "Icon"

    # medium horizontal rects → buttons
    if 1.2 < aspect < 6 and 20 < h < 55 and 40 < w < 250:
        return "Button"

    # medium rect near top → tabs
    if 1.5 < aspect < 5 and 20 < h < 50 and 50 < w < 200 and cy < 0.15:
        return "Tab"

    # larger blocks → image cards
    if area_pct > 0.01 and w > 80 and h > 80:
        return "Image/Card"

    # really large → containers/sections
    if area_pct > 0.05:
        return "Container"

    # horizontal text-ish blocks
    if aspect > 1.5 and w > 50 and h > 10:
        return "Link/Text Block"

    return None


def detect_html_components(image):
    """Run the full HTML component detection pipeline on a BGR image.
    Returns (annotated_image, list_of_detections)."""

    annotated = image.copy()
    img_h, img_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # adaptive threshold picks up UI element edges really well
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )

    # morph ops to merge nearby fragments into coherent regions
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    dilated = cv2.dilate(thresh, kh, iterations=1)
    dilated = cv2.dilate(dilated, kv, iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kc, iterations=2)

    # second pass with canny to catch stuff the threshold missed
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.dilate(edges, kc, iterations=2)

    combined = cv2.bitwise_or(closed, edges)

    contours, _ = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    seen = []  # for dedup
    MIN_AREA = 400

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # skip the whole image or hairline-thin stuff
        if w > img_w * 0.95 and h > img_h * 0.95:
            continue
        if w < 5 or h < 5:
            continue

        # overlap check — skip duplicates
        dup = False
        for (sx, sy, sw, sh) in seen:
            ox1, oy1 = max(x, sx), max(y, sy)
            ox2, oy2 = min(x + w, sx + sw), min(y + h, sy + sh)
            if ox1 < ox2 and oy1 < oy2:
                overlap = (ox2 - ox1) * (oy2 - oy1)
                if overlap / min(w * h, sw * sh) > 0.6:
                    dup = True
                    break
        if dup:
            continue

        comp = classify_component(x, y, w, h, img_w, img_h)
        if comp is None:
            continue

        seen.append((x, y, w, h))
        color = COMPONENT_COLORS.get(comp, (255, 255, 255))

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # draw label above the box
        (tw, th), _ = cv2.getTextSize(comp, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = max(y - 2, th + 6)
        cv2.rectangle(annotated, (x, ly - th - 6), (x + tw + 8, ly + 2), color, -1)
        cv2.putText(annotated, comp, (x + 4, ly - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        detections.append({
            "Component": comp,
            "X": x, "Y": y,
            "Width": w, "Height": h,
            "Confidence": "Heuristic"
        })

    return annotated, detections
