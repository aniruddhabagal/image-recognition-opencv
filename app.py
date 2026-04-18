import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os

st.set_page_config(
    page_title="Screenshot Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# inject custom styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-title {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(135deg, #7C3AED, #EC4899, #F59E0B);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.2rem; letter-spacing: -0.5px;
}
.hero-sub { font-size: 1rem; color: #94A3B8; margin-bottom: 1.5rem; }

.glass-card {
    background: rgba(124, 58, 237, 0.08); border: 1px solid rgba(124, 58, 237, 0.2);
    border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; backdrop-filter: blur(10px);
}

.metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
.metric-card {
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(236,72,153,0.1));
    border: 1px solid rgba(124,58,237,0.25); border-radius: 12px;
    padding: 1rem 1.4rem; flex: 1; text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(124,58,237,0.15); }
.metric-val { font-size: 2rem; font-weight: 700; color: #A78BFA; }
.metric-lbl { font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem; }

[data-testid="stSidebarContent"] { background: linear-gradient(180deg, #0F0F1A 0%, #1A1A2E 100%); }

.mode-desc {
    background: rgba(124, 58, 237, 0.06); border-left: 3px solid #7C3AED;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin-top: 0.5rem;
    font-size: 0.85rem; color: #CBD5E1;
}

.badge {
    display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.5px;
    background: rgba(245, 158, 11, 0.15); color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.legend-item {
    display: inline-flex; align-items: center; gap: 0.4rem;
    margin-right: 1rem; margin-bottom: 0.4rem; font-size: 0.8rem; color: #CBD5E1;
}
.legend-dot { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }

[data-testid="stFileUploader"] { border: 2px dashed rgba(124,58,237,0.3); border-radius: 12px; padding: 0.5rem; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
.element-container { animation: fadeIn 0.3s ease-in; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)


# --- mode config ---
MODES = {
    "🧩 Detect HTML Components": "Identifies UI elements by name — Buttons, Input Fields, Nav Bars, Images, Icons, Tabs. Uses contour analysis + heuristic classification.",
    "🎯 Detect Objects (YOLOv8)": "Detects real-world objects (person, car, laptop, phone…) using YOLOv8 trained on 80 COCO classes.",
    "📝 Detect Text (OCR)":       "Finds and reads all text in the image using EasyOCR. Shows recognized text, position, and confidence.",
    "✏️ Detect Edges & Contours": "Detects structural edges and shape boundaries with Canny edge detection. Classifies shapes (rectangles, circles, etc.).",
}


# --- sidebar ---
with st.sidebar:
    st.markdown('<p class="hero-title">🔍 Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Screenshot Component Detection</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### 🎛️ Detection Mode")
    mode = st.radio("Pick a mode:", list(MODES.keys()), index=0, label_visibility="collapsed")
    st.markdown(f'<div class="mode-desc">{MODES[mode]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📤 Upload Image")
    uploaded = st.file_uploader("Upload", type=["png", "jpg", "jpeg", "bmp", "webp"], label_visibility="collapsed")
    use_sample = st.checkbox("Use sample image (image.png)", value=False)

    st.markdown("---")
    st.markdown("#### ⚙️ Parameters")

    if "Object" in mode:
        conf_thresh = st.slider("Confidence Threshold", 0.05, 0.95, 0.15, 0.05)
    elif "Text" in mode:
        min_text_conf = st.slider("Min Text Confidence", 0.0, 0.9, 0.1, 0.05)
    elif "Edge" in mode:
        min_cont_area = st.slider("Min Contour Area", 50, 2000, 300, 50)
    elif "HTML" in mode:
        sensitivity = st.select_slider("Detection Sensitivity", options=["Low", "Medium", "High"], value="Medium")

    st.markdown("---")
    st.markdown('<div style="text-align:center; color:#64748B; font-size:0.75rem;">Streamlit · OpenCV · YOLOv8 · EasyOCR</div>', unsafe_allow_html=True)


# --- main area ---
st.markdown('<p class="hero-title">Screenshot Component Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload a screenshot and pick a detection mode to analyze it</p>', unsafe_allow_html=True)

# load image
image = None
if uploaded is not None:
    raw = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
elif use_sample:
    path = os.path.join(os.path.dirname(__file__), "image.png")
    if os.path.exists(path):
        image = cv2.imread(path)
    else:
        st.error("image.png not found in the project folder.")

if image is None:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:4rem 2rem;">
        <div style="font-size:4rem; margin-bottom:1rem;">📸</div>
        <div style="font-size:1.2rem; color:#A78BFA; font-weight:600; margin-bottom:0.5rem;">No Image Loaded</div>
        <div style="color:#64748B;">Upload a screenshot in the sidebar or check "Use sample image" to get started.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# --- run detection ---
st.markdown(f'<span class="badge">⏳ PROCESSING — {mode}</span>', unsafe_allow_html=True)
prog = st.progress(0, text="Loading detector...")
t0 = time.time()

try:
    if "HTML" in mode:
        prog.progress(20, text="Analyzing UI structure...")
        from detectors.html_components import detect_html_components
        annotated, detections = detect_html_components(image)
        cols = ["Component", "X", "Y", "Width", "Height"]

    elif "Object" in mode:
        prog.progress(20, text="Running YOLOv8...")
        from detectors.objects import detect_objects
        annotated, detections = detect_objects(image, conf=conf_thresh)
        cols = ["Object", "Confidence", "X1", "Y1", "X2", "Y2"]

    elif "Text" in mode:
        prog.progress(20, text="Running OCR (first run downloads models)...")
        from detectors.text_ocr import detect_text
        annotated, detections = detect_text(image)
        cols = ["Text", "Confidence", "X1", "Y1", "X2", "Y2"]

    elif "Edge" in mode:
        prog.progress(20, text="Detecting edges...")
        from detectors.edges import detect_edges
        annotated, detections = detect_edges(image, min_area=min_cont_area)
        cols = ["Shape", "Area", "Vertices", "X", "Y", "Width", "Height"]

    else:
        st.error("Unknown mode selected.")
        st.stop()

    elapsed = time.time() - t0
    prog.progress(100, text=f"Done in {elapsed:.1f}s!")
    time.sleep(0.4)
    prog.empty()

except Exception as e:
    prog.empty()
    st.error(f"Detection failed: {e}")
    st.exception(e)
    st.stop()


# --- results ---
n = len(detections)

if "HTML" in mode:
    unique = len(set(d["Component"] for d in detections))
    ulabel = "Component Types"
elif "Object" in mode:
    unique = len(set(d["Object"] for d in detections))
    ulabel = "Object Classes"
elif "Text" in mode:
    unique = n
    ulabel = "Text Regions"
else:
    unique = len(set(d["Shape"] for d in detections))
    ulabel = "Shape Types"

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card"><div class="metric-val">{n}</div><div class="metric-lbl">Detections</div></div>
    <div class="metric-card"><div class="metric-val">{unique}</div><div class="metric-lbl">{ulabel}</div></div>
    <div class="metric-card"><div class="metric-val">{elapsed:.1f}s</div><div class="metric-lbl">Time</div></div>
    <div class="metric-card"><div class="metric-val">{image.shape[1]}×{image.shape[0]}</div><div class="metric-lbl">Resolution</div></div>
</div>
""", unsafe_allow_html=True)

# color legend for HTML mode
if "HTML" in mode:
    from detectors.html_components import COMPONENT_COLORS
    html = '<div style="margin-bottom:1rem;">'
    for name, bgr in COMPONENT_COLORS.items():
        hx = '#{:02x}{:02x}{:02x}'.format(bgr[2], bgr[1], bgr[0])
        html += f'<span class="legend-item"><span class="legend-dot" style="background:{hx}"></span>{name}</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# side-by-side images
c1, c2 = st.columns(2)
with c1:
    st.markdown("##### 📷 Original")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
with c2:
    st.markdown(f"##### 🔍 Detected — {mode}")
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

# data tables
st.markdown("##### 📊 Details")

if detections:
    df = pd.DataFrame(detections)

    if "HTML" in mode:
        summary = df["Component"].value_counts().reset_index()
        summary.columns = ["Component", "Count"]
        ca, cb = st.columns([1, 2])
        with ca:
            st.markdown("**Summary**")
            st.dataframe(summary, use_container_width=True, hide_index=True)
        with cb:
            st.markdown("**All Detections**")
            st.dataframe(df[cols], use_container_width=True, hide_index=True)

    elif "Object" in mode:
        summary = df["Object"].value_counts().reset_index()
        summary.columns = ["Object", "Count"]
        ca, cb = st.columns([1, 2])
        with ca:
            st.markdown("**Summary**")
            st.dataframe(summary, use_container_width=True, hide_index=True)
        with cb:
            st.markdown("**All Detections**")
            st.dataframe(df[cols], use_container_width=True, hide_index=True)
    else:
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
else:
    st.info("No detections found. Try adjusting the parameters.")

st.markdown("---")
st.markdown('<div style="text-align:center; color:#475569; font-size:0.8rem;">Screenshot Component Detection Tool</div>', unsafe_allow_html=True)
