# Selective Object Detection (YOLOv8)

A focused object-detection demo for interview submission.

## What is special

- Mode-based filtering: detects only the selected category.
- Includes a dedicated mode: **Detect HTML Components**.
- Streamlit frontend for interactive demo.
- CLI mode for quick reproducible runs.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run Frontend Demo

```powershell
streamlit run app.py
```

Then:

1. Upload an image.
2. Select detection mode (for example, Detect HTML Components).
3. Click **Run Detection**.

## Run CLI

### Persons only

```powershell
python main.py --image image.png --preset "Detect Persons" --no-show
```

### HTML components only

```powershell
python main.py --image image.png --preset "Detect HTML Components" --no-show
```

### All classes

```powershell
python main.py --image image.png --preset "Detect Everything" --no-show
```

## Presets

- Detect Everything
- Detect HTML Components
- Detect Persons
- Detect Vehicles
- Detect Animals
- Detect Electronics

## Notes

- HTML mode is mapped to closest YOLO COCO classes: `tv`, `laptop`, `keyboard`, `mouse`, `cell phone`, `remote`, `book`.
- If you need true UI element detection (button/input/div/etc.), train or fine-tune a UI-focused dataset/model.
