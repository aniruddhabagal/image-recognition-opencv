import cv2
import numpy as np
import pandas as pd
import streamlit as st

from main import PRESET_DEFINITIONS, detect_and_annotate, load_model, resolve_target_classes


st.set_page_config(page_title="Selective Object Detection", page_icon="OD", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --bg-a: #f7f5ef;
        --bg-b: #f2e6d2;
        --ink: #1f1b16;
        --accent: #cc4f16;
        --accent-soft: #f9d8c6;
        --panel: rgba(255, 250, 240, 0.82);
        --line: #d8bca2;
    }

    .stApp {
        background:
            radial-gradient(1200px 550px at 10% -10%, #f9ceb4 0%, transparent 55%),
            radial-gradient(900px 450px at 100% 0%, #f3dfbb 0%, transparent 62%),
            linear-gradient(145deg, var(--bg-a), var(--bg-b));
        color: var(--ink);
    }

    h1, h2, h3 {
        font-family: 'Sora', sans-serif !important;
        letter-spacing: -0.02em;
        color: var(--ink);
    }

    p, label, div, span {
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    .hero {
        border: 1px solid var(--line);
        background: var(--panel);
        backdrop-filter: blur(4px);
        padding: 1.15rem 1.35rem;
        border-radius: 16px;
        box-shadow: 0 12px 28px rgba(74, 42, 13, 0.08);
        margin-bottom: 1rem;
        animation: fadeInUp 620ms ease-out;
    }

    .metric-card {
        border: 1px dashed var(--line);
        background: rgba(255, 247, 236, 0.8);
        border-radius: 12px;
        padding: 0.7rem 0.9rem;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stButton > button {
        border-radius: 999px;
        border: none;
        background: linear-gradient(90deg, #b93f11, #d45d18);
        color: #fff;
        font-weight: 700;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 8px 18px rgba(185, 63, 17, 0.24);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        filter: brightness(1.05);
    }

    .block-container {
        padding-top: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1 style="margin:0;">Selective Object Detection Demo</h1>
        <p style="margin:0.4rem 0 0 0;">
            Pick one detection mode and the model will return only that category.
            Example: <b>Detect HTML Components</b> shows only screen-like components.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model("yolov8n.pt")


model = get_model()
all_class_names = sorted(model.names.values())

with st.sidebar:
    st.header("Detection Controls")
    mode_options = list(PRESET_DEFINITIONS.keys()) + ["Custom"]
    selected_mode = st.selectbox("Mode", mode_options, index=0)

    custom_classes = []
    if selected_mode == "Custom":
        custom_classes = st.multiselect(
            "Select one or more classes",
            options=all_class_names,
            default=["person"],
        )

    confidence = st.slider("Confidence", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
    iou = st.slider("IoU", min_value=0.10, max_value=0.90, value=0.45, step=0.05)

    st.caption("Tip: Higher confidence gives fewer but cleaner detections.")

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])
run_btn = st.button("Run Detection")

if not uploaded:
    st.info("Upload an image to begin.")
elif run_btn:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not decode image. Try another file.")
    else:
        target_classes = resolve_target_classes(selected_mode, custom_classes)
        annotated_rgb, detections = detect_and_annotate(
            model=model,
            image_bgr=image_bgr,
            target_class_names=target_classes,
            conf=confidence,
            iou=iou,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader(f"Detected ({selected_mode})")
            st.image(annotated_rgb, use_container_width=True)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Objects found", len(detections))
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            if detections:
                table = pd.DataFrame(detections)
                st.dataframe(table, use_container_width=True)
            else:
                st.warning("No objects matched this mode for the current image.")
else:
    st.info("Click 'Run Detection' to process the uploaded image.")
