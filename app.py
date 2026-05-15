import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image
import numpy as np
import time
from collections import Counter, deque
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PPE AI Shield Pro",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS (Enhanced for interactivity) ──────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Barlow:wght@400;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background: #0a0c0f;
    color: #e0e6ef;
  }

  [data-testid="stSidebar"] { background: #0f1318 !important; border-right: 1px solid #1e2a38; }
  
  /* Glassmorphism effect for metrics */
  [data-testid="metric-container"] {
    background: rgba(17, 24, 32, 0.8);
    border: 1px solid #1e2a38;
    border-radius: 12px;
    padding: 1rem !important;
    backdrop-filter: blur(10px);
  }

  .section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4d7fa8;
    margin: 1.5rem 0 0.5rem;
    border-bottom: 1px solid #1e2a38;
  }

  .alert-danger { background: #2d0a0a; border-left: 4px solid #ff3b3b; padding: 1rem; border-radius: 4px; color: #ff7a7a; font-weight: bold; }
  .alert-ok { background: #0a2d1a; border-left: 4px solid #00e5a0; padding: 1rem; border-radius: 4px; color: #00e5a0; }

  /* Hide Streamlit branding */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Replace with your actual path
    return YOLO("runs/detect/ppe_detection_v25/weights/best.pt")

try:
    model = load_model()
except:
    st.error("Model file not found. Please check the path.")
    st.stop()

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0
if "alert_count" not in st.session_state:
    st.session_state.alert_count = 0
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0

# ─── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def calculate_safety_score(violations, total_objects):
    if total_objects == 0: return 100
    score = 100 - (violations / total_objects * 100)
    return max(0, round(score, 1))

def plot_safety_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "SITE SAFETY SCORE", 'font': {'size': 14, 'color': '#4d7fa8'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4d7fa8"},
            'bar': {'color': "#00e5a0"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#1e2a38",
            'steps': [
                {'range': [0, 50], 'color': '#3d0a0a'},
                {'range': [50, 80], 'color': '#3d2d0a'},
                {'range': [80, 100], 'color': '#0a2d1a'}
            ],
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#e0e6ef", 'family': "IBM Plex Mono"})
    return fig

def apply_per_class_filter(results, model_names, base_conf, violation_conf):
    CLASS_THRESHOLDS = {
        "no_helmet": violation_conf, "no_vest": violation_conf,
        "helmet": base_conf, "vest": base_conf, "gloves": base_conf,
        "boots": base_conf, "goggles": base_conf, "mask": base_conf,
    }
    boxes = results[0].boxes
    if not boxes or len(boxes) == 0: return [], []
    
    filtered_labels, filtered_confs = [], []
    for cls_id, conf_val in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        label = model_names[int(cls_id)]
        threshold = CLASS_THRESHOLDS.get(label, base_conf)
        if conf_val >= threshold:
            filtered_labels.append(label)
            filtered_confs.append(conf_val)
    return filtered_labels, filtered_confs

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='color:#00e5a0;'>🛡️ PPE SHIELD</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.6rem; color:#4d7fa8;'>PRO VERSION v2.5</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<p class='section-header'>⚙️ Detection Config</p>", unsafe_allow_html=True)
    confidence = st.slider("Confidence", 0.10, 1.00, 0.50, 0.05)
    iou_thresh = st.slider("IoU (NMS)", 0.10, 1.00, 0.40, 0.05)
    violation_conf = st.slider("Violation Threshold", 0.30, 1.00, 0.65, 0.05)

    st.markdown("<p class='section-header'>🎞 Temporal Smoothing</p>", unsafe_allow_html=True)
    smoothing_window = st.slider("Frame Buffer", 1, 10, 5)
    smoothing_votes = st.slider("Min Frames to Alert", 1, 10, 3)

    st.markdown("<p class='section-header'>🎛 Display</p>", unsafe_allow_html=True)
    live_mode = st.toggle("Auto Run Detection", True)
    show_labels = st.toggle("Show Labels", True)
    show_conf = st.toggle("Show Confidence", True)
    show_boxes = st.toggle("Show Boxes", True)

# ─── MAIN HEADER ───────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("<h1 style='margin:0;'>Real-time Safety Monitor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#4d7fa8;'>Automated PPE Compliance System</p>", unsafe_allow_html=True)

with col_h2:
    # The Dynamic Gauge lives here
    # We'll calculate the score based on current session history
    all_violations = sum(h['violations'] for h in st.session_state.history)
    all_objs = sum(h['objects'] for h in st.session_state.history)
    current_score = calculate_safety_score(all_violations, all_objs)
    st.plotly_chart(plot_safety_gauge(current_score), use_container_width=True)

st.divider()

# ─── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "🎥 Video Stream", "📊 Intelligence Log"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE (INTERACTIVE)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        
        # Interactivity: Let user filter which labels to focus on
        st.markdown("<p class='section-header'>Detection Settings</p>", unsafe_allow_html=True)
        
        if live_mode:
            with st.spinner("Analyzing..."):
                results = model(image, conf=min(confidence, violation_conf), iou=iou_thresh)
                detected_labels, confs = apply_per_class_filter(results, model.names, confidence, violation_conf)
                
                # Update Stats
                st.session_state.scan_count += 1
                st.session_state.total_detections += len(detected_labels)
                violations = sum(1 for l in detected_labels if l.startswith("no_"))
                if violations > 0: st.session_state.alert_count += 1
                
                st.session_state.history.append({
                    "scan_id": st.session_state.scan_count,
                    "file": uploaded.name,
                    "objects": len(detected_labels),
                    "violations": violations,
                    "classes": dict(Counter(detected_labels)),
                    "timestamp": time.strftime("%H:%M:%S")
                })

                # Display Image and Results
                res_img = results[0].plot(labels=show_labels, conf=show_conf, boxes=show_boxes)
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.image(res_img, use_container_width=True, caption="Processed Frame")
                
                with c2:
                    st.markdown("### Detection Summary")
                    if violations > 0:
                        st.markdown(f"<div class='alert-danger'>⚠️ {violations} VIOLATIONS</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='alert-ok'>✅ COMPLIANT</div>", unsafe_allow_html=True)
                    
                    # Interactive Class Filter
                    all_unique_classes = list(set(detected_labels))
                    selected_classes = st.multiselect("Filter Class Analytics", all_unique_classes, default=all_unique_classes)

                    # Dynamic Chart (Plotly)
                    if selected_classes:
                        chart_data = Counter({k: v for k, v in Counter(detected_labels).items() if k in selected_classes})
                        if chart_data:
                            df_chart = pd.DataFrame(chart_data.items(), columns=['Class', 'Count'])
                            fig = px.bar(df_chart, x='Class', y='Count', color='Class', 
                                         color_discrete_sequence=px.colors.sequential.Greens_r)
                            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                             font_color="white", margin=dict(l=10, r=10, t=10, b=10), height=250)
                            st.plotly_chart(fig, use_container_width=True)

        else:
            st.image(image, use_container_width=True)
            st.info("Enable 'Auto Run' in sidebar to perform detection.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO (DYNAMIC FEED)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.flush()

        if live_mode:
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            progress_bar = st.progress(0)
            
            # Layout for Live Dashboard
            dash_col1, dash_col2 = st.columns([3, 1])
            with dash_col2:
                st.markdown("<p class='section-header'>Live Alert Log</p>", unsafe_allow_html=True)
                alert_log = st.empty()
                log_entries = []

            violation_buffer = deque(maxlen=smoothing_window)
            frame_idx = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                frame_idx += 1
                # Skip frames for performance
                if frame_idx % 3 != 0: continue

                results = model(frame, conf=min(confidence, violation_conf), verbose=False)
                annotated = results[0].plot(labels=show_labels, conf=show_conf, boxes=show_boxes)
                
                labels, _ = apply_per_class_filter(results, model.names, confidence, violation_conf)
                has_violation = any(l.startswith("no_") for l in labels)
                violation_buffer.append(has_violation)
                
                # Display Video
                stframe.image(annotated, channels="BGR", use_container_width=True)
                progress_bar.progress(frame_idx / total_frames)

                # Dynamic Alert Logic
                if sum(violation_buffer) >= smoothing_votes:
                    msg = "🚨 CRITICAL: PPE VIOLATION"
                    color = "danger"
                    log_entries.insert(0, f"[{time.strftime('%H:%M:%S')}] {msg}")
                elif has_violation:
                    msg = "⏳ WARNING: Unconfirmed Violation"
                    color = "warning"
                    log_entries.insert(0, f"[{time.strftime('%H:%M:%S')}] {msg}")
                else:
                    msg = "✅ System Clear"
                    color = "ok"
                    log_entries.insert(0, f"[{time.strftime('%H:%M:%S')}] {msg}")
                
                # Limit log size for performance
                alert_log.markdown(f"<div style='font-family:monospace; font-size:0.8rem;'>{'<br>'.join(log_entries[:10])}</div>", unsafe_allow_html=True)

            cap.release()
            st.success("Video Processing Complete")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY (INTERACTIVE DATAFRAME)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<p class='section-header'>Intelligence & Compliance Log</p>", unsafe_allow_html=True)
    
    if st.session_state.history:
        # Convert history to DataFrame for powerful interactivity
        df_history = pd.DataFrame(st.session_state.history)
        
        # Flatten the 'classes' dictionary for the dataframe
        class_df = df_history['classes'].apply(pd.Series).fillna(0)
        df_final = pd.concat([df_history.drop(columns=['classes']), class_df], axis=1)

        # Interactive controls
        col_a, col_b = st.columns(2)
        with col_a:
            search = st.text_input("🔍 Search by Filename", "")
        with col_b:
            if st.button("🗑️ Clear All History"):
                st.session_state.history = []
                st.rerun()

        # Apply Search Filter
        if search:
            df_final = df_final[df_final['file'].str.contains(search, case=False)]

        st.dataframe(
            df_final, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "scan_id": "ID",
                "file": "File Name",
                "objects": "Total Obj",
                "violations": "Violations",
                "timestamp": "Time"
            }
        )

        # Export
        st.download_button(
            "📥 Download Compliance Report (JSON)",
            data=json.dumps(st.session_state.history, indent=2),
            file_name="ppe_report.json",
            mime="application/json"
        )
    else:
        st.info("No detection data available. Start a scan to populate the log.")

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<br><hr><center><small style='color:#2a3f55;'>PPE AI SHIELD • YOLOv8s • REAL-TIME MONITORING ENGINE</small></center>", unsafe_allow_html=True)