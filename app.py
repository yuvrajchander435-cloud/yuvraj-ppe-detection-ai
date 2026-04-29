import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image
import numpy as np
import time
from collections import Counter

# ------------------ CONFIG ------------------
st.set_page_config(page_title="PPE AI Dashboard", layout="wide")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return YOLO("runs/detect/ppe_detection_v25/weights/best.pt")

model = load_model()

# ------------------ SESSION STATE ------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Control Panel")

confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3)
live_mode = st.sidebar.toggle("⚡ Auto Run Detection", True)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 About")
st.sidebar.info("AI-powered PPE Detection using YOLOv8s")

# ------------------ HEADER ------------------
st.markdown("""
<h1 style='text-align: center; color: #00FFAA;'>🦺 PPE Detection Dashboard</h1>
""", unsafe_allow_html=True)

# ------------------ TABS ------------------
tab1, tab2 = st.tabs(["📷 Image Detection", "🎥 Video Detection"])

# ------------------ IMAGE TAB ------------------
with tab1:
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        if live_mode:
            with st.spinner("Detecting..."):
                results = model(image, conf=confidence)
                result_img = results[0].plot()

                boxes = results[0].boxes
                classes = boxes.cls.tolist() if boxes else []
                names = model.names

                detected_labels = [names[int(cls)] for cls in classes]
                count = Counter(detected_labels)

                # Store history
                st.session_state.history.append(count)

            with col2:
                st.image(result_img, caption="Detection Output", use_container_width=True)

            # -------- STATS --------
            st.markdown("### 📊 Detection Insights")

            colA, colB, colC = st.columns(3)
            colA.metric("Total Objects", len(classes))
            colB.metric("Unique Classes", len(count))
            colC.metric("Confidence", confidence)

            # -------- CLASS BREAKDOWN --------
            st.markdown("### 🧾 Class Distribution")
            st.bar_chart(count)

            # -------- ALERT SYSTEM --------
            if "no_helmet" in count:
                st.error("🚨 ALERT: Worker without helmet detected!")

# ------------------ VIDEO TAB ------------------
with tab2:
    video_file = st.file_uploader("Upload Video", type=["mp4"])

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        st.video(tfile.name)

        if live_mode:
            cap = cv2.VideoCapture(tfile.name)

            stframe = st.empty()
            progress = st.progress(0)

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current = 0
            start_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=confidence)
                annotated = results[0].plot()

                stframe.image(annotated, channels="BGR", use_container_width=True)

                current += 1
                progress.progress(min(current / frame_count, 1.0))

            cap.release()

            end_time = time.time()
            fps = frame_count / (end_time - start_time)

            st.success(f"✅ Done! Avg FPS: {fps:.2f}")

# ------------------ HISTORY ------------------
st.markdown("### 🧠 Detection History")

if st.session_state.history:
    st.write(st.session_state.history[-5:])
else:
    st.info("No history yet.")

# ------------------ FOOTER ------------------
st.markdown("""
<hr>
<center>🚀 Smart PPE Monitoring System | Built with Streamlit + YOLOv8</center>
""", unsafe_allow_html=True)