"""
TRACK THE UNTRACKABLE: Configuration Module
Long-Horizon Multi-Object Identity Recovery Under Camera, Occlusion & Appearance Drift
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


@dataclass
class DetectorConfig:
    """YOLO Object Detector settings."""
    model_name: str = "yolov8n.pt"  # Lightweight YOLOv8 nano by default
    conf_threshold: float = 0.35    # Confidence cutoff for detections
    nms_iou_threshold: float = 0.50 # NMS threshold
    target_classes: List[int] = field(default_factory=lambda: [0])  # Class 0: Person (COCO)
    device: str = "auto"            # "auto", "cuda", "cpu"
    half_precision: bool = False    # FP16 if supported on GPU


@dataclass
class ReidConfig:
    """Appearance Re-Identification feature extractor settings."""
    backbone: str = "mobilenet_v3"   # Fast and lightweight or "resnet50"
    embedding_dim: int = 512         # Deep feature embedding dimension
    input_size: Tuple[int, int] = (256, 128)  # Height, Width standard for ReID
    use_color_hist: bool = True      # Fuse deep embeddings with HSV color histogram
    color_hist_weight: float = 0.20  # Weight for color histogram vs deep embedding
    device: str = "auto"


@dataclass
class CameraMotionConfig:
    """Global Motion Compensation (GMC) settings."""
    enabled: bool = True
    method: str = "sparse_optical_flow"  # "sparse_optical_flow" or "orb"
    max_features: int = 400
    ransac_threshold: float = 3.0
    min_inliers: int = 15


@dataclass
class IdentityMemoryConfig:
    """Long-horizon appearance and track memory configuration."""
    max_history_frames: int = 600      # Keep lost tracks in memory for up to 600 frames (~20s at 30fps)
    gallery_capacity_per_id: int = 8   # Max representative feature vectors per ID
    ema_alpha: float = 0.85            # Exponential Moving Average weight for running feature template
    reid_recovery_threshold: float = 0.68  # Cosine similarity threshold for recovering identity
    spatial_consistency_weight: float = 0.25 # Weight for spatial proximity during recovery
    temporal_decay_half_life: int = 200 # Frames after which spatial gating weight decays


@dataclass
class TrackerConfig:
    """Hierarchical Multi-Object Tracker settings."""
    track_high_thresh: float = 0.50    # First-stage matching confidence threshold
    track_low_thresh: float = 0.15     # Second-stage matching confidence threshold
    new_track_thresh: float = 0.60     # Minimum confidence to initialize a new track
    track_buffer: int = 60             # Frames before a lost track is sent to long-horizon memory
    match_iou_thresh: float = 0.70     # Spatial IoU threshold for stage 1 matching
    unconfirmed_iou_thresh: float = 0.50 # Spatial IoU for stage 2 low-confidence matching
    reid_distance_weight: float = 0.40 # Balance between IoU distance and ReID cosine distance
    gmc_enabled: bool = True           # Enable camera ego-motion compensation
    enable_identity_recovery: bool = True # Enable long-horizon identity memory recovery


@dataclass
class VisualizerConfig:
    """Visualization and stream rendering settings."""
    draw_trajectories: bool = True
    trajectory_length: int = 50
    draw_camera_motion: bool = True
    draw_occlusion_boxes: bool = True
    draw_reid_similarity: bool = True
    box_thickness: int = 2
    font_scale: float = 0.55
    # Distinct bright colors for IDs (BGR format for OpenCV)
    palette: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (0, 255, 128),   # Emerald Green
        (255, 128, 0),   # Deep Azure
        (0, 165, 255),   # Bright Orange
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Yellow
        (255, 255, 0),   # Cyan
        (180, 105, 255), # Hot Pink
        (50, 205, 50),   # Lime Green
        (238, 130, 238), # Violet
        (0, 69, 255),    # Red-Orange
        (255, 191, 0),   # Deep Sky
        (205, 92, 92),   # Medium Coral
    ])


@dataclass
class AppConfig:
    """Master application configuration."""
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    reid: ReidConfig = field(default_factory=ReidConfig)
    camera_motion: CameraMotionConfig = field(default_factory=CameraMotionConfig)
    memory: IdentityMemoryConfig = field(default_factory=IdentityMemoryConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)
    output_dir: str = "output"
    sample_videos_dir: str = "sample_videos"
