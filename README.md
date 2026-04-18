# Screenshot Component Detection Tool

A visual analysis tool that detects and classifies different types of components in screenshots. Upload any screenshot and the tool will identify UI elements, objects, text, or structural edges depending on the selected mode.

Built with Streamlit, OpenCV, YOLOv8, and EasyOCR.

## Features

- **HTML Component Detection** — Identifies buttons, input fields, navigation bars, images, icons, tabs, and other UI elements by analyzing their visual structure. Each component is labeled by its type.
- **Object Detection (YOLOv8)** — Uses a pretrained YOLOv8 model to detect real-world objects (people, cars, laptops, etc.) across 80 COCO classes.
- **Text Detection (OCR)** — Extracts and reads all text from the screenshot with bounding boxes and confidence scores using EasyOCR.
- **Edge & Contour Detection** — Finds structural boundaries and classifies shapes (rectangles, circles, polygons) using Canny edge detection.

Each detection mode runs **independently** — selecting one mode only runs that specific detector, not all of them.

## How It Works

### HTML Component Detection
Uses a two-pass approach combining adaptive thresholding and Canny edge detection to find contours in the image. Contours are then classified into component types based on their dimensions, aspect ratio, and position:
- Full-width thin bands → Navigation Bar
- Wide horizontal rectangles → Input/Search Fields
- Small squared elements → Icons
- Medium horizontal rectangles → Buttons
- Large blocks → Image Cards, Containers

### Object Detection
Runs YOLOv8n (nano) for fast inference. Confidence threshold is adjustable from the sidebar.

### Text Detection
EasyOCR handles both detection and recognition. The bounding box color reflects confidence (green = high, red = low).

### Edge Detection
Auto-thresholds Canny parameters using the image's median intensity. Contours are approximated with `approxPolyDP` and classified by vertex count.

## Setup

```bash
# create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# install dependencies
pip install -r requirements.txt
```

> **Note:** EasyOCR downloads language models (~100MB) on first use. YOLOv8 weights (`yolov8n.pt`) are auto-downloaded on first run.

## Usage

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Use the sidebar to:
1. Pick a detection mode
2. Upload an image (or check "Use sample image")
3. Adjust detection parameters
4. View results — side-by-side comparison + data table

## Project Structure

```
├── app.py                  # Streamlit frontend
├── detectors/
│   ├── __init__.py
│   ├── html_components.py  # contour-based UI element classifier
│   ├── objects.py          # YOLOv8 wrapper
│   ├── text_ocr.py         # EasyOCR wrapper
│   └── edges.py            # Canny + contour shape detection
├── .streamlit/
│   └── config.toml         # dark theme config
├── image.png               # sample screenshot for testing
├── requirements.txt
└── main.py                 # original standalone detection script
```

## Tech Stack

| Tool | Purpose |
|------|---------|
| Streamlit | Web UI framework |
| OpenCV | Image processing, contour analysis |
| YOLOv8 (ultralytics) | Object detection |
| EasyOCR | Text detection & recognition |
| NumPy, Pandas | Data handling |

