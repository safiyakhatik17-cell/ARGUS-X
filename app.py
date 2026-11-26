"""
TRACK THE UNTRACKABLE: Streamlit Dashboard
Long-Horizon Multi-Object Identity Recovery Under Camera, Occlusion & Appearance Drift
"""

import os
import sys
import time
import tempfile
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Import core modules
from config import AppConfig, TrackerConfig, DetectorConfig, ReidConfig, CameraMotionConfig, IdentityMemoryConfig, VisualizerConfig
from detection import YOLODetector, Detection
from tracking import (
    LongHorizonTracker,
    Track,
    TrackObservation,
    merge_bidirectional_observations,
)
from reid import ReIDExtractor
from camera_motion import CameraMotionCompensator
from identity_memory import IdentityMemory
from visualizer import TrackVisualizer
from benchmark_generator import generate_benchmark_video


# Page Configuration
st.set_page_config(
    page_title="Track the Untrackable",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek Dark Theme
st.markdown("""
<style>
    .main {
        background-color: #0b0e14;
    }
    .metric-card {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 13px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .highlight-magenta {
        color: #f778ba;
    }
    .highlight-green {
        color: #3fb950;
    }
    .highlight-yellow {
        color: #d29922;
    }
    .stProgress > div > div > div > div {
        background-color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initializes persistent session state."""
    if "processed_video_path" not in st.session_state:
        st.session_state.processed_video_path = None
    if "tracking_telemetry" not in st.session_state:
        st.session_state.tracking_telemetry = []
    if "identity_summaries" not in st.session_state:
        st.session_state.identity_summaries = []
    if "recovery_events" not in st.session_state:
        st.session_state.recovery_events = []
    if "benchmark_path" not in st.session_state:
        st.session_state.benchmark_path = None


init_session_state()

# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/radar.png", width=64)
st.sidebar.title("ARGUS-X / Untrackable")
st.sidebar.caption("Long-Horizon Multi-Object Identity Recovery")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Pipeline Settings")

# 1. Detection Settings
with st.sidebar.expander("Object Detection (YOLO)", expanded=False):
    det_conf = st.slider("Detection Confidence", 0.15, 0.90, 0.35, 0.05)
    det_nms = st.slider("NMS IoU Threshold", 0.20, 0.80, 0.45, 0.05)
    target_class_choice = st.selectbox("Target Classes", ["Person Only (COCO 0)", "All Objects"])
    target_classes = [0] if "Person" in target_class_choice else None

# 2. Camera Ego-Motion Compensation (GMC)
with st.sidebar.expander("Camera Motion Compensation", expanded=True):
    gmc_enabled = st.checkbox("Enable GMC (Optical Flow)", value=True)
    gmc_method = st.selectbox("Method", ["sparse_optical_flow", "orb"], index=0)
    gmc_features = st.slider("Max Background Features", 100, 800, 350, 50)

# 3. Long-Horizon Identity Memory (ReID)
with st.sidebar.expander("Identity Memory & ReID", expanded=True):
    enable_recovery = st.checkbox("Enable Long-Horizon Recovery", value=True)
    bidirectional_tracking = st.checkbox(
        "Track forward and backward",
        value=True,
        help="Process the video in both directions and reconcile track IDs before exporting."
    )
    reid_thresh = st.slider("Recovery Cosine Threshold", 0.50, 0.90, 0.68, 0.02)
    max_memory_frames = st.slider("Memory Retention (Frames)", 60, 1200, 600, 60)
    ema_weight = st.slider("Appearance EMA Alpha", 0.50, 0.95, 0.85, 0.05)
    spatial_weight = st.slider("Spatial Extrapolation Weight", 0.0, 0.50, 0.25, 0.05)

# 4. Visualization Overlays
with st.sidebar.expander("Visual Overlays", expanded=False):
    draw_trails = st.checkbox("Motion Trajectory Trails", value=True)
    draw_compass = st.checkbox("Camera Ego-Motion Compass", value=True)
    draw_occlusion_box = st.checkbox("Occlusion Position Predictions", value=True)
    trail_len = st.slider("Trail History Length", 10, 100, 45, 5)

# -------------------------------------------------------------
# Main Hero Header
# -------------------------------------------------------------
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown("""
    <h1 style='color: #58a6ff; margin-bottom: 2px;'>TRACK THE UNTRACKABLE</h1>
    <h4 style='color: #8b949e; font-weight: 400;'>Long-Horizon Multi-Object Identity Recovery Under Camera, Occlusion & Appearance Drift</h4>
    """, unsafe_allow_html=True)
with col_status:
    st.markdown("""
    <div style='text-align: right; padding-top: 15px;'>
        <span style='background-color: #238636; color: white; padding: 6px 12px; border-radius: 20px; font-size: 13px; font-weight: 600;'>
            ● ENGINE READY
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------------------
# Video Source Selection
# -------------------------------------------------------------
source_tab1, source_tab2 = st.tabs(["📁 Upload Video File", "🧪 Synthetic Occlusion & Camera Drift Benchmark"])

video_file_path: Optional[str] = None

with source_tab1:
    uploaded_file = st.file_uploader(
        "Upload Video (.mp4, .avi, .mov, .mkv)",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a pedestrian or surveillance video to track targets and recover identities."
    )
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        video_file_path = tfile.name
        st.success(f"Loaded: `{uploaded_file.name}`")

with source_tab2:
    st.markdown("""
    **Synthetic Benchmark Generator**:
    Simulates severe challenges in real-world computer vision:
    - Dynamic camera pan/drift (ego-motion).
    - Pedestrians with distinct clothing appearances.
    - Central architectural pillar causing a **35-frame complete occlusion**.
    - Re-emergence identity recovery verification.
    """)
    if st.button("Generate Synthetic Benchmark Video", type="secondary"):
        with st.spinner("Rendering synthetic benchmark sequence..."):
            bench_path = "sample_videos/benchmark_occlusion_cam_drift.mp4"
            os.makedirs("sample_videos", exist_ok=True)
            generate_benchmark_video(bench_path, num_frames=120)
            st.session_state.benchmark_path = bench_path
            st.success("Benchmark video generated successfully!")

    if st.session_state.benchmark_path and os.path.exists(st.session_state.benchmark_path):
        st.info("Using generated synthetic benchmark video.")
        video_file_path = st.session_state.benchmark_path

st.markdown("---")

# -------------------------------------------------------------
# Video Processing Execution
# -------------------------------------------------------------
if video_file_path:
    # Open video to read metadata
    cap = cv2.VideoCapture(video_file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    with col_meta1:
        st.metric("Total Frames", f"{total_frames}")
    with col_meta2:
        st.metric("Resolution", f"{width_in} x {height_in}")
    with col_meta3:
        st.metric("Native FPS", f"{fps_in:.1f}")
    with col_meta4:
        max_process_frames = st.number_input("Process Frame Limit", min_value=10, max_value=max(10, total_frames), value=min(300, total_frames), step=25)

    btn_col1, btn_col2 = st.columns([2, 5])
    with btn_col1:
        start_processing = st.button("🚀 Process Video", type="primary", use_container_width=True)

    if start_processing:
        st.session_state.processed_video_path = None
        # Construct pipeline configs
        det_cfg = DetectorConfig(
            conf_threshold=det_conf,
            nms_iou_threshold=det_nms,
            target_classes=target_classes or [0]
        )
        reid_cfg = ReidConfig()
        gmc_cfg = CameraMotionConfig(
            enabled=gmc_enabled,
            method=gmc_method,
            max_features=gmc_features
        )
        mem_cfg = IdentityMemoryConfig(
            max_history_frames=max_memory_frames,
            reid_recovery_threshold=reid_thresh,
            ema_alpha=ema_weight,
            spatial_consistency_weight=spatial_weight
        )
        trk_cfg = TrackerConfig(
            track_high_thresh=det_conf,
            gmc_enabled=gmc_enabled,
            enable_identity_recovery=enable_recovery
        )
        vis_cfg = VisualizerConfig(
            draw_trajectories=draw_trails,
            trajectory_length=trail_len,
            draw_camera_motion=draw_compass,
            draw_occlusion_boxes=draw_occlusion_box
        )

        # Initialize components
        detector = YOLODetector(det_cfg)
        reid_extractor = ReIDExtractor(reid_cfg)
        camera_comp = CameraMotionCompensator(gmc_cfg)
        identity_memory = IdentityMemory(mem_cfg)
        tracker = LongHorizonTracker(
            tracker_config=trk_cfg,
            reid_extractor=reid_extractor,
            camera_compensator=camera_comp,
            identity_memory=identity_memory
        )
        reverse_tracker = None
        if bidirectional_tracking:
            reverse_tracker = LongHorizonTracker(
                tracker_config=trk_cfg,
                reid_extractor=reid_extractor,
                camera_compensator=CameraMotionCompensator(gmc_cfg),
                identity_memory=IdentityMemory(mem_cfg)
            )
        visualizer = TrackVisualizer(vis_cfg)

        # Streamlit Live UI components
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        preview_placeholder = st.empty()

        cap = cv2.VideoCapture(video_file_path)
        detections_by_frame = []
        forward_observations: List[List[TrackObservation]] = []
        forward_camera_stats: List[Dict[str, float]] = []
        start_time = time.time()
        frames_to_run = min(total_frames, int(max_process_frames))

        def capture_tracks(tracks: List[Track], source_frame_idx: int) -> List[TrackObservation]:
            return [
                TrackObservation(
                    frame_idx=source_frame_idx + 1,
                    track_id=track.track_id,
                    bbox=tuple(float(value) for value in track.bbox),
                    score=track.score,
                    state=track.state,
                    is_recovered=track.is_recovered,
                    recovery_similarity=track.recovery_similarity,
                    time_since_update=track.time_since_update
                )
                for track in tracks
            ]

        # Forward pass also caches detections so the reverse pass does not rerun YOLO.
        while cap.isOpened() and len(detections_by_frame) < frames_to_run:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            detections = detector.detect(frame)
            active_tracks = tracker.update(frame, detections)
            detections_by_frame.append(detections)
            forward_observations.append(capture_tracks(active_tracks, len(detections_by_frame) - 1))
            forward_camera_stats.append(camera_comp.last_motion_stats.copy())

            if len(detections_by_frame) % 3 == 0 or len(detections_by_frame) == frames_to_run:
                progress = 0.5 * len(detections_by_frame) / max(1, frames_to_run)
                progress_bar.progress(progress)
                status_text.text(
                    f"Forward pass: {len(detections_by_frame)}/{frames_to_run} frames | "
                    f"Active Tracks: {len(active_tracks)}"
                )

        cap.release()
        processed_frames = len(detections_by_frame)
        if processed_frames == 0:
            st.error("The selected video could not be decoded; no frames were processed.")
        else:
            backward_observations: List[List[TrackObservation]] = [
                [] for _ in range(processed_frames)
            ]
            if reverse_tracker is not None:
                reverse_cap = cv2.VideoCapture(video_file_path)
                for source_frame_idx in range(processed_frames - 1, -1, -1):
                    reverse_cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)
                    ret, frame = reverse_cap.read()
                    if not ret or frame is None:
                        reverse_cap.release()
                        raise RuntimeError(
                            f"Could not decode frame {source_frame_idx + 1} for reverse tracking."
                        )

                    reverse_tracks = reverse_tracker.update(
                        frame, detections_by_frame[source_frame_idx]
                    )
                    backward_observations[source_frame_idx] = capture_tracks(
                        reverse_tracks, source_frame_idx
                    )

                    if source_frame_idx % 3 == 0 or source_frame_idx == 0:
                        completed = processed_frames - source_frame_idx
                        progress = 0.5 + 0.25 * completed / processed_frames
                        progress_bar.progress(progress)
                        status_text.text(
                            f"Backward pass: {completed}/{processed_frames} frames | "
                            f"Active Tracks: {len(reverse_tracks)}"
                        )
                reverse_cap.release()

            observations_by_frame = merge_bidirectional_observations(
                forward_observations, backward_observations
            )
            total_recoveries = tracker.total_recoveries
            if reverse_tracker is not None:
                total_recoveries += reverse_tracker.total_recoveries

            # Render the reconciled tracks in original playback order.
            os.makedirs("output", exist_ok=True)
            out_video_path = os.path.join("output", "processed_tracked_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(out_video_path, fourcc, fps_in, (width_in, height_in))
            if not out_writer.isOpened():
                st.error("Could not create the processed video output.")
            else:
                render_cap = cv2.VideoCapture(video_file_path)
                telemetry: List[Dict[str, Any]] = []
                trajectories: Dict[int, List[Tuple[float, float]]] = {}
                processing_fps = processed_frames / max(1e-3, time.time() - start_time)

                for source_frame_idx, frame_observations in enumerate(observations_by_frame):
                    ret, frame = render_cap.read()
                    if not ret or frame is None:
                        render_cap.release()
                        out_writer.release()
                        raise RuntimeError(
                            f"Could not decode frame {source_frame_idx + 1} while rendering."
                        )

                    visible_tracks = []
                    for observation in frame_observations:
                        x1, y1, x2, y2 = observation.bbox
                        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        history = trajectories.setdefault(observation.track_id, [])
                        history.append(center)

                        rendered_track = Track(
                            observation.track_id,
                            list(observation.bbox),
                            observation.score
                        )
                        rendered_track.state = observation.state
                        rendered_track.is_recovered = observation.is_recovered
                        rendered_track.recovery_similarity = observation.recovery_similarity
                        rendered_track.time_since_update = observation.time_since_update
                        rendered_track.trajectory = history[-trail_len:]
                        visible_tracks.append(rendered_track)

                        forward_camera = forward_camera_stats[source_frame_idx]
                        telemetry.append({
                            "frame": observation.frame_idx,
                            "track_id": observation.track_id,
                            "x1": round(x1, 1),
                            "y1": round(y1, 1),
                            "x2": round(x2, 1),
                            "y2": round(y2, 1),
                            "confidence": round(observation.score, 3),
                            "state": observation.state.name,
                            "is_recovered": observation.is_recovered,
                            "recovery_score": round(observation.recovery_similarity, 3)
                                if observation.is_recovered else 0.0,
                            "camera_dx": round(forward_camera.get("dx", 0.0), 2),
                            "camera_dy": round(forward_camera.get("dy", 0.0), 2)
                        })

                    rendered_frame = visualizer.render_frame(
                        frame=frame,
                        tracks=visible_tracks,
                        camera_stats=forward_camera_stats[source_frame_idx],
                        frame_idx=source_frame_idx + 1,
                        fps=processing_fps,
                        total_recoveries=total_recoveries
                    )
                    out_writer.write(rendered_frame)

                    if (source_frame_idx + 1) % 3 == 0 or source_frame_idx + 1 == processed_frames:
                        progress = 0.75 + 0.25 * (source_frame_idx + 1) / processed_frames
                        progress_bar.progress(progress)
                        status_text.text(
                            f"Rendering merged output: {source_frame_idx + 1}/{processed_frames} frames"
                        )
                        preview_rgb = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)
                        preview_placeholder.image(
                            preview_rgb, channels="RGB", use_container_width=True
                        )

                render_cap.release()
                out_writer.release()
                total_time = max(1e-3, time.time() - start_time)
                avg_fps = processed_frames / total_time

                st.session_state.processed_video_path = out_video_path
                st.session_state.tracking_telemetry = telemetry
                st.session_state.identity_summaries = identity_memory.get_identity_summary()
                st.session_state.recovery_events = identity_memory.recovery_events

                st.success(
                    f"Processing Complete! Processed {processed_frames} frames in "
                    f"{total_time:.2f}s (Average FPS: {avg_fps:.1f})"
                )

# -------------------------------------------------------------
# Results, Metrics, Gallery & Downloads
# -------------------------------------------------------------
if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
    st.markdown("---")
    st.subheader("📊 Tracking Analytics & Recovery Metrics")

    telemetry_data = st.session_state.tracking_telemetry
    summaries = st.session_state.identity_summaries
    recoveries = st.session_state.recovery_events

    total_unique_ids = len({row["track_id"] for row in telemetry_data})
    total_recovered_count = sum(bool(row["is_recovered"]) for row in telemetry_data)
    prevented_id_switches = total_recovered_count

    # Metric Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Unique IDs</div>
            <div class='metric-value'>{total_unique_ids}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Recovered Identities</div>
            <div class='metric-value highlight-magenta'>{total_recovered_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>ID Switches Prevented</div>
            <div class='metric-value highlight-green'>{prevented_id_switches}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        max_occ = max([r.get("occlusion_duration_frames", 0) for r in recoveries]) if recoveries else 0
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Max Occlusion Recovered</div>
            <div class='metric-value highlight-yellow'>{max_occ} <span style='font-size:16px;'>frames</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Detections Logged</div>
            <div class='metric-value'>{len(telemetry_data)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for Detailed Analytics
    tab_video, tab_timeline, tab_gallery, tab_telemetry = st.tabs([
        "🎬 Processed Video & Download",
        "📈 Identity Lifecycle Timeline",
        "👤 Identity Memory Gallery",
        "📋 Telemetry Data & Export"
    ])

    with tab_video:
        st.markdown("### Processed Tracking Output")
        v_col1, v_col2 = st.columns([3, 1])
        with v_col1:
            st.video(st.session_state.processed_video_path)
        with v_col2:
            st.markdown("#### Export Artifacts")
            with open(st.session_state.processed_video_path, "rb") as vf:
                video_bytes = vf.read()
                st.download_button(
                    label="⬇️ Download Processed Video (.mp4)",
                    data=video_bytes,
                    file_name="tracked_untrackable_output.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

            if telemetry_data:
                df_telemetry = pd.DataFrame(telemetry_data)
                csv_bytes = df_telemetry.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Telemetry (.csv)",
                    data=csv_bytes,
                    file_name="tracking_telemetry.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    with tab_timeline:
        st.markdown("### Identity Timeline & Occlusion Survival")
        if telemetry_data:
            df = pd.DataFrame(telemetry_data)
            fig = px.scatter(
                df,
                x="frame",
                y="track_id",
                color="state",
                symbol="is_recovered",
                title="Identity Longevity: Continuous Tracking & Re-emergence After Occlusions",
                labels={"frame": "Frame Number", "track_id": "Global Identity ID"},
                color_discrete_map={
                    "Tracked": "#3fb950",
                    "Occluded": "#d29922",
                    "New": "#58a6ff"
                },
                template="plotly_dark",
                height=380
            )
            fig.update_traces(marker=dict(size=7))
            fig.update_layout(yaxis=dict(type='category'))
            st.plotly_chart(fig, use_container_width=True)

    with tab_gallery:
        st.markdown("### Long-Horizon Memory Bank Gallery")
        if summaries:
            cols = st.columns(min(4, len(summaries)))
            for i, summ in enumerate(summaries):
                with cols[i % len(cols)]:
                    st.markdown(f"""
                    <div style='background-color:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px; margin-bottom:12px;'>
                        <h4 style='color:#58a6ff; margin:0;'>ID #{summ['track_id']}</h4>
                        <p style='color:#8b949e; font-size:12px; margin:4px 0;'>State: <b>{summ['state'].upper()}</b></p>
                        <p style='color:#8b949e; font-size:12px; margin:4px 0;'>Active Frames: <b>{summ['total_detections']}</b></p>
                        <p style='color:#8b949e; font-size:12px; margin:4px 0;'>Recoveries: <b style='color:#f778ba;'>{summ['recoveries']}</b></p>
                        <p style='color:#8b949e; font-size:12px; margin:4px 0;'>Lifespan: Frames {summ['first_frame']} → {summ['last_frame']}</p>
                    </div>
                    """, unsafe_allow_html=True)

    with tab_telemetry:
        st.markdown("### Telemetry Data Log")
        if telemetry_data:
            df = pd.DataFrame(telemetry_data)
            st.dataframe(df, use_container_width=True, height=350)
