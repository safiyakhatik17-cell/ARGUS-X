"""
Visualization Module for Track the Untrackable
Renders glowing bounding boxes, persistent IDs, recovery badges,
motion trajectory trails, occlusion predictions, and camera ego-motion HUD.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

from config import VisualizerConfig
from tracking import Track, TrackState


class TrackVisualizer:
    """
    Renders high-visibility overlays for multi-object tracking,
    re-identification recovery, and camera ego-motion telemetry.
    """

    def __init__(self, config: Optional[VisualizerConfig] = None):
        self.config = config or VisualizerConfig()
        self.palette = self.config.palette

    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        """Returns distinct BGR color for given track ID."""
        idx = (track_id - 1) % len(self.palette)
        return self.palette[idx]

    def render_frame(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        camera_stats: Optional[Dict[str, float]] = None,
        frame_idx: int = 0,
        fps: float = 0.0,
        total_recoveries: int = 0
    ) -> np.ndarray:
        """
        Draws tracking overlays, trajectory history, badges, and telemetry HUD on frame.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 1. Draw Trajectories
        if self.config.draw_trajectories:
            for track in tracks:
                color = self.get_color(track.track_id)
                pts = track.trajectory[-self.config.trajectory_length:]
                if len(pts) >= 2:
                    for i in range(1, len(pts)):
                        pt1 = (int(round(pts[i - 1][0])), int(round(pts[i - 1][1])))
                        pt2 = (int(round(pts[i][0])), int(round(pts[i][1])))
                        # Fade trajectory alpha towards the past
                        alpha = float(i) / float(len(pts))
                        thickness = max(1, int(round(2 * alpha)))
                        cv2.line(vis, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

        # 2. Draw Bounding Boxes and Identity Badges
        for track in tracks:
            color = self.get_color(track.track_id)
            x1, y1, x2, y2 = [int(round(v)) for v in track.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            is_occluded = (track.state == TrackState.Occluded)
            is_recovered = track.is_recovered

            if is_occluded and not self.config.draw_occlusion_boxes:
                continue

            # Box styling
            if is_recovered:
                # Glowing Gold/Magenta border for recovered identity
                box_color = (255, 0, 255) # Magenta highlight
                cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, self.config.box_thickness + 2)
                cv2.rectangle(vis, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 1)
            elif is_occluded:
                # Dashed yellow/amber line for occluded / predicted track
                box_color = (0, 215, 255) # Amber
                self._draw_dashed_rect(vis, (x1, y1), (x2, y2), box_color, 2)
            else:
                # Standard vibrant box
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, self.config.box_thickness)

            # Label composition
            label_parts = [f"ID {track.track_id}"]
            if is_recovered:
                label_parts.append(f"RECOVERED ({track.recovery_similarity:.2f})")
            elif is_occluded:
                label_parts.append(f"OCCLUDED [t+{track.time_since_update}]")
            else:
                label_parts.append(f"{track.score:.2f}")

            label_text = " | ".join(label_parts)
            
            # Draw label background
            (txt_w, txt_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, 1
            )
            lbl_y1 = max(0, y1 - txt_h - 8)
            lbl_y2 = y1
            lbl_x2 = min(w, x1 + txt_w + 10)

            badge_bg = (20, 20, 20) if not is_recovered else (80, 0, 80)
            cv2.rectangle(vis, (x1, lbl_y1), (lbl_x2, lbl_y2), badge_bg, -1)
            
            # Badge text
            txt_color = (0, 255, 255) if is_recovered else (255, 255, 255)
            cv2.putText(
                vis, label_text, (x1 + 5, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, txt_color, 1, cv2.LINE_AA
            )

        # 3. Telemetry HUD (Top Banner)
        self._draw_hud(vis, camera_stats, frame_idx, fps, total_recoveries, len(tracks))

        return vis

    def _draw_hud(
        self,
        img: np.ndarray,
        camera_stats: Optional[Dict[str, float]],
        frame_idx: int,
        fps: float,
        total_recoveries: int,
        active_count: int
    ):
        """Draws top information bar and camera ego-motion compass."""
        h, w = img.shape[:2]

        # Top banner background (semi-transparent dark bar)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (15, 17, 23), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

        # Title and frame stats
        title = "TRACK THE UNTRACKABLE | LONG-HORIZON IDENTITY RECOVERY"
        cv2.putText(img, title, (15, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        stats_str = f"Frame: {frame_idx} | FPS: {fps:.1f} | Active Tracks: {active_count} | Recovered Identities: {total_recoveries}"
        cv2.putText(img, stats_str, (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        # Camera Ego-Motion Compass (Upper right corner)
        if self.config.draw_camera_motion and camera_stats:
            cx_hub, cy_hub = w - 60, 21
            cv2.circle(img, (cx_hub, cy_hub), 14, (60, 60, 60), 1, cv2.LINE_AA)
            cv2.circle(img, (cx_hub, cy_hub), 2, (200, 200, 200), -1)

            dx = camera_stats.get("dx", 0.0)
            dy = camera_stats.get("dy", 0.0)
            mag = np.sqrt(dx * dx + dy * dy)

            # Draw camera pan vector
            if mag > 0.5:
                # Scale vector for visibility
                scale = min(12.0, mag * 2.0)
                end_x = int(round(cx_hub - (dx / mag) * scale)) # Pan compensation direction
                end_y = int(round(cy_hub - (dy / mag) * scale))
                cv2.arrowedLine(img, (cx_hub, cy_hub), (end_x, end_y), (0, 255, 128), 2, tipLength=0.35)

            cam_str = f"GMC: dx={dx:+.1f} dy={dy:+.1f}"
            cv2.putText(img, cam_str, (w - 220, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 128), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_dashed_rect(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 1):
        """Draws dashed rectangle for predicted occluded targets."""
        x1, y1 = pt1
        x2, y2 = pt2
        dash_len = 6

        # Top and bottom
        for x in range(x1, x2, dash_len * 2):
            cv2.line(img, (x, y1), (min(x + dash_len, x2), y1), color, thickness)
            cv2.line(img, (x, y2), (min(x + dash_len, x2), y2), color, thickness)

        # Left and right
        for y in range(y1, y2, dash_len * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_len, y2)), color, thickness)
            cv2.line(img, (x2, y), (x2, min(y + dash_len, y2)), color, thickness)
