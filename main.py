import argparse
from typing import Dict, List, Optional

import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO


PRESET_DEFINITIONS: Dict[str, Optional[List[str]]] = {
    "Detect Everything": None,
    "Detect HTML Components": [
        "tv",
        "laptop",
        "keyboard",
        "mouse",
        "cell phone",
        "remote",
        "book",
    ],
    "Detect Persons": ["person"],
    "Detect Vehicles": [
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "train",
        "airplane",
        "boat",
    ],
    "Detect Animals": [
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ],
    "Detect Electronics": [
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
    ],
}


def load_model(weights_path: str = "yolov8n.pt") -> YOLO:
    return YOLO(weights_path)


def class_names_to_ids(model: YOLO, class_names: List[str]) -> List[int]:
    name_to_id = {name: idx for idx, name in model.names.items()}
    return [name_to_id[name] for name in class_names if name in name_to_id]


def resolve_target_classes(
    preset_name: str,
    custom_class_names: Optional[List[str]] = None,
) -> Optional[List[str]]:
    if preset_name == "Custom":
        return custom_class_names or []
    return PRESET_DEFINITIONS.get(preset_name)


def detect_and_annotate(
    model: YOLO,
    image_bgr,
    target_class_names: Optional[List[str]] = None,
    conf: float = 0.25,
    iou: float = 0.45,
):
    class_ids = None
    if target_class_names is not None:
        class_ids = class_names_to_ids(model, target_class_names)
        if not class_ids:
            # No valid class names for this model; skip model call to avoid false positives.
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), []

    result = model(image_bgr, conf=conf, iou=iou, classes=class_ids)[0]
    output_rgb = cv2.cvtColor(image_bgr.copy(), cv2.COLOR_BGR2RGB)
    detections = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        score = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        label = f"{class_name} {score:.2f}"

        cv2.rectangle(output_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_label_top = max(y1 - th - 6, 0)
        cv2.rectangle(output_rgb, (x1, y_label_top), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(
            output_rgb,
            label,
            (x1 + 2, max(y1 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        detections.append(
            {
                "class": class_name,
                "confidence": round(score, 4),
                "bbox": [x1, y1, x2, y2],
            }
        )

    return output_rgb, detections


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 filtered object detection")
    parser.add_argument("--image", default="image.png", help="Path to input image")
    parser.add_argument("--weights", default="yolov8n.pt", help="Path to model weights")
    parser.add_argument(
        "--preset",
        default="Detect Everything",
        choices=list(PRESET_DEFINITIONS.keys()),
        help="Detection mode",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--save", default="", help="Optional output image path")
    parser.add_argument("--no-show", action="store_true", help="Do not show matplotlib preview")
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    model = load_model(args.weights)
    selected_classes = resolve_target_classes(args.preset)
    annotated_rgb, detections = detect_and_annotate(
        model=model,
        image_bgr=image_bgr,
        target_class_names=selected_classes,
        conf=args.conf,
        iou=args.iou,
    )

    print(f"Mode: {args.preset}")
    print(f"Detected {len(detections)} objects")
    if detections:
        counts = {}
        for item in detections:
            counts[item["class"]] = counts.get(item["class"], 0) + 1
        print("Counts by class:")
        for class_name, count in sorted(counts.items()):
            print(f"  - {class_name}: {count}")

    if args.save:
        cv2.imwrite(args.save, cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved annotated image to: {args.save}")

    if not args.no_show:
        plt.figure(figsize=(16, 9))
        plt.imshow(annotated_rgb)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    run_cli()