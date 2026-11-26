"""
Synthetic Benchmark Video Generator
Creates realistic evaluation sequences simulating moving camera pan/drift,
moving pedestrians with distinctive appearance clothing, and severe long-horizon occlusions.
"""

import os
from typing import Tuple, List, Dict, Any
import numpy as np
import cv2


def generate_benchmark_video(
    output_path: str = "sample_videos/benchmark_occlusion_cam_drift.mp4",
    num_frames: int = 150,
    width: int = 800,
    height: int = 480,
    fps: int = 25
) -> str:
    """
    Renders a realistic synthetic video demonstrating:
    1. Camera pan / ego-motion drift from left to right (background motion).
    2. Person A (distinctive Cyan shirt + Dark Navy pants) moving left to right.
    3. Person B (distinctive Coral Red shirt + Olive pants) moving right to left.
    4. Large central architectural pillar causing a 35-frame complete occlusion of Person A.
    5. Re-emergence testing identity recovery under camera drift!
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Background texture generation (grid of urban tiles and buildings)
    bg_width = width + 600
    bg = np.zeros((height, bg_width, 3), dtype=np.uint8)
    
    # Gradient pavement / street
    for y in range(height):
        ratio = y / height
        val = int(40 + 70 * ratio)
        bg[y, :, :] = (val, val + 5, val + 10)

    # Road markers / sidewalk texture for optical flow to lock onto
    for x in range(0, bg_width, 60):
        cv2.line(bg, (x, int(height * 0.4)), (x, height), (70, 75, 80), 1)
        cv2.circle(bg, (x + 20, int(height * 0.7)), 4, (120, 120, 130), -1)
        cv2.circle(bg, (x + 45, int(height * 0.85)), 5, (90, 95, 100), -1)

    # Wall/storefront in background
    cv2.rectangle(bg, (0, 0), (bg_width, int(height * 0.35)), (60, 65, 75), -1)
    for bx in range(30, bg_width - 50, 140):
        cv2.rectangle(bg, (bx, 30), (bx + 90, int(height * 0.32)), (90, 100, 115), -1)
        cv2.putText(bg, "SHOP", (bx + 20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 170, 185), 1)

    # Central obstacle (Concrete Pillar / Billboard)
    pillar_w = 120
    pillar_x_world = 550
    pillar_y1 = int(height * 0.15)
    pillar_y2 = int(height * 0.95)

    # Person dimensions
    p_w, p_h = 42, 105

    for f in range(num_frames):
        # Camera panning ego-motion: pans 2.2 pixels rightward per frame
        cam_x_offset = int(f * 2.2)
        frame_bg = bg[:, cam_x_offset:cam_x_offset + width].copy()

        # Person A (Cyan shirt, Navy pants): moves left to right in world coordinates
        # World x starts at 220, moves +4.2 px/frame
        pA_world_x = 220 + f * 4.2
        pA_cam_x = int(pA_world_x - cam_x_offset)
        pA_y = int(height * 0.52)

        # Person B (Coral Red shirt, Olive pants): moves right to left
        pB_world_x = 1000 - f * 3.6
        pB_cam_x = int(pB_world_x - cam_x_offset)
        pB_y = int(height * 0.50)

        # Draw Person A (behind pillar)
        # Check if Person A is within frame boundaries
        if -p_w < pA_cam_x < width:
            _draw_synthetic_person(
                frame_bg, pA_cam_x, pA_y, p_w, p_h,
                shirt_color=(235, 180, 50), # Cyan / Azure in BGR
                pants_color=(120, 40, 20),  # Dark Navy Blue in BGR
                head_color=(180, 205, 230), frame_idx=f
            )

        # Draw Person B (behind pillar)
        if -p_w < pB_cam_x < width:
            _draw_synthetic_person(
                frame_bg, pB_cam_x, pB_y, p_w, p_h,
                shirt_color=(40, 40, 220),  # Bright Coral Red in BGR
                pants_color=(40, 120, 60),  # Olive Green in BGR
                head_color=(170, 195, 220), frame_idx=f
            )

        # Draw Foreground Occlusion Pillar on top of pedestrians
        pillar_cam_x = int(pillar_x_world - cam_x_offset)
        if -pillar_w < pillar_cam_x < width:
            px1 = max(0, pillar_cam_x)
            px2 = min(width, pillar_cam_x + pillar_w)
            # Metallic Concrete pillar gradient
            cv2.rectangle(frame_bg, (px1, pillar_y1), (px2, pillar_y2), (45, 50, 55), -1)
            cv2.rectangle(frame_bg, (px1, pillar_y1), (px2, pillar_y2), (90, 100, 110), 2)
            cv2.putText(
                frame_bg, "PILLAR", (px1 + 15, int((pillar_y1 + pillar_y2) / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 150, 165), 2
            )
            cv2.putText(
                frame_bg, "[OCCLUSION ZONE]", (px1 + 5, int((pillar_y1 + pillar_y2) / 2) + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 165, 255), 1
            )

        out.write(frame_bg)

    out.release()
    return output_path


def _draw_synthetic_person(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    shirt_color: Tuple[int, int, int],
    pants_color: Tuple[int, int, int],
    head_color: Tuple[int, int, int],
    frame_idx: int
):
    """Draws a recognizable stylized human figure with head, torso, and legs."""
    ih, iw = img.shape[:2]
    
    # Head
    head_r = int(w * 0.28)
    head_cx = x + w // 2
    head_cy = y + head_r + 2
    if 0 <= head_cx < iw and 0 <= head_cy < ih:
        cv2.circle(img, (head_cx, head_cy), head_r, head_color, -1)

    # Torso (Shirt)
    torso_y1 = head_cy + head_r
    torso_y2 = torso_y1 + int(h * 0.42)
    tx1 = max(0, min(iw - 1, x + 4))
    tx2 = max(0, min(iw, x + w - 4))
    ty1 = max(0, min(ih - 1, torso_y1))
    ty2 = max(0, min(ih, torso_y2))
    if tx2 > tx1 and ty2 > ty1:
        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), shirt_color, -1)

    # Legs (Pants) with subtle walking stride motion
    leg_y1 = torso_y2
    leg_y2 = y + h
    leg_w = (w - 10) // 2
    stride = int(np.sin(frame_idx * 0.4) * 4)

    # Left leg
    lx1 = max(0, min(iw - 1, x + 4 - stride))
    lx2 = max(0, min(iw, lx1 + leg_w))
    ly1 = max(0, min(ih - 1, leg_y1))
    ly2 = max(0, min(ih, leg_y2))
    if lx2 > lx1 and ly2 > ly1:
        cv2.rectangle(img, (lx1, ly1), (lx2, ly2), pants_color, -1)

    # Right leg
    rx1 = max(0, min(iw - 1, x + 6 + leg_w + stride))
    rx2 = max(0, min(iw, rx1 + leg_w))
    if rx2 > rx1 and ly2 > ly1:
        cv2.rectangle(img, (rx1, ly1), (rx2, ly2), pants_color, -1)
