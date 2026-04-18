import cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt

# Load YOLOv8 nano model (fast + accurate, auto-downloads on first run)
model = YOLO("yolov8n.pt")

img = cv2.imread("image.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run detection — lower conf threshold to catch small/faint objects
results = model(img, conf=0.25, iou=0.45)[0]

# Draw bounding boxes and labels on the image
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    cls_id = int(box.cls[0])
    label = f"{model.names[cls_id]} {conf:.2f}"

    # Draw rectangle
    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img_rgb, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
    cv2.putText(img_rgb, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

print(f"Detected {len(results.boxes)} objects")

plt.figure(figsize=(16, 9))
plt.imshow(img_rgb)
plt.axis("off")
plt.tight_layout()
plt.show()