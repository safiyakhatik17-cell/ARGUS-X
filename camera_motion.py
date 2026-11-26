"""
Camera Motion Compensation (Global Motion Compensation - GMC)
Compensates for camera pan, tilt, rotation, and zoom to decouple ego-motion from object motion.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

from config import CameraMotionConfig


class CameraMotionCompensator:
    """
    Estimates global camera motion between consecutive frames using sparse optical flow
    or feature matching with RANSAC affine transformation.
    """

    def __init__(self, config: Optional[CameraMotionConfig] = None):
        self.config = config or CameraMotionConfig()
        self.prev_frame_gray: Optional[np.ndarray] = None
        self.prev_keypoints: Optional[np.ndarray] = None
        self.current_transform: np.ndarray = np.eye(2, 3, dtype=np.float32)
        self.cumulative_transform: np.ndarray = np.eye(3, 3, dtype=np.float32)
        
        # Diagnostics
        self.last_inlier_count: int = 0
        self.last_motion_stats: Dict[str, float] = {"dx": 0.0, "dy": 0.0, "scale": 1.0, "rotation_deg": 0.0}

    def reset(self):
        """Reset motion history for a new video sequence."""
        self.prev_frame_gray = None
        self.prev_keypoints = None
        self.current_transform = np.eye(2, 3, dtype=np.float32)
        self.cumulative_transform = np.eye(3, 3, dtype=np.float32)
        self.last_inlier_count = 0
        self.last_motion_stats = {"dx": 0.0, "dy": 0.0, "scale": 1.0, "rotation_deg": 0.0}

    def compute_motion(
        self,
        frame: np.ndarray,
        exclusion_boxes: Optional[List[List[float]]] = None
    ) -> np.ndarray:
        """
        Computes 2x3 affine transformation matrix M from frame (t-1) to frame (t).
        
        Args:
            frame: Current BGR or Grayscale frame.
            exclusion_boxes: List of [x1, y1, x2, y2] to mask out detected moving objects,
                             ensuring optical flow only tracks static background.
                             
        Returns:
            2x3 Affine transformation matrix M such that:
            [x_new, y_new]^T = M * [x_old, y_old, 1]^T
        """
        if not self.config.enabled:
            return np.eye(2, 3, dtype=np.float32)

        # Convert to grayscale
        if len(frame.shape) == 3:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = frame.copy()

        h, w = curr_gray.shape

        if self.prev_frame_gray is None:
            self.prev_frame_gray = curr_gray
            self.current_transform = np.eye(2, 3, dtype=np.float32)
            return self.current_transform

        # Create mask excluding foreground objects (pedestrians, vehicles)
        mask = np.ones((h, w), dtype=np.uint8) * 255
        if exclusion_boxes:
            for box in exclusion_boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 0

        # Feature detection and tracking
        if self.config.method == "sparse_optical_flow":
            transform = self._optical_flow_gmc(self.prev_frame_gray, curr_gray, mask)
        else:
            transform = self._orb_gmc(self.prev_frame_gray, curr_gray, mask)

        self.current_transform = transform
        self._update_cumulative_and_metrics(transform)
        self.prev_frame_gray = curr_gray

        return self.current_transform

    def _optical_flow_gmc(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Computes motion using Shi-Tomasi corners and Lucas-Kanade optical flow."""
        # Detect good features to track in previous frame
        pts_prev = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.config.max_features,
            qualityLevel=0.01,
            minDistance=10,
            mask=mask,
            blockSize=5
        )

        if pts_prev is None or len(pts_prev) < self.config.min_inliers:
            self.last_inlier_count = 0
            return np.eye(2, 3, dtype=np.float32)

        # Track keypoints into current frame
        pts_curr, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            pts_prev,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        valid_idx = (status.flatten() == 1)
        matched_prev = pts_prev[valid_idx]
        matched_curr = pts_curr[valid_idx]

        if len(matched_prev) < self.config.min_inliers:
            self.last_inlier_count = len(matched_prev)
            return np.eye(2, 3, dtype=np.float32)

        # Robust affine transformation estimation with RANSAC
        M, inliers = cv2.estimateAffinePartial2D(
            matched_prev,
            matched_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_threshold,
            maxIters=1000
        )

        if M is None or inliers is None or np.sum(inliers) < self.config.min_inliers:
            self.last_inlier_count = 0
            return np.eye(2, 3, dtype=np.float32)

        self.last_inlier_count = int(np.sum(inliers))
        return M.astype(np.float32)

    def _orb_gmc(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Computes motion using ORB feature detection and brute force matching."""
        orb = cv2.ORB_create(nfeatures=self.config.max_features)
        kp1, des1 = orb.detectAndCompute(prev_gray, mask)
        kp2, des2 = orb.detectAndCompute(curr_gray, mask)

        if des1 is None or des2 is None or len(kp1) < self.config.min_inliers or len(kp2) < self.config.min_inliers:
            self.last_inlier_count = 0
            return np.eye(2, 3, dtype=np.float32)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) < self.config.min_inliers:
            self.last_inlier_count = len(matches)
            return np.eye(2, 3, dtype=np.float32)

        pts_prev = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_curr = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, inliers = cv2.estimateAffinePartial2D(
            pts_prev,
            pts_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_threshold
        )

        if M is None or inliers is None or np.sum(inliers) < self.config.min_inliers:
            self.last_inlier_count = 0
            return np.eye(2, 3, dtype=np.float32)

        self.last_inlier_count = int(np.sum(inliers))
        return M.astype(np.float32)

    def _update_cumulative_and_metrics(self, M: np.ndarray):
        """Calculates dx, dy, zoom, and rotation metrics from transformation matrix."""
        # M is [[a, -b, tx], [b, a, ty]]
        a = M[0, 0]
        b = M[1, 0]
        tx = M[0, 2]
        ty = M[1, 2]

        scale = float(np.sqrt(a * a + b * b))
        angle_rad = float(np.arctan2(b, a))
        angle_deg = float(np.degrees(angle_rad))

        self.last_motion_stats = {
            "dx": float(tx),
            "dy": float(ty),
            "scale": scale,
            "rotation_deg": angle_deg
        }

        # Update 3x3 cumulative matrix
        M_3x3 = np.eye(3, dtype=np.float32)
        M_3x3[:2, :] = M
        self.cumulative_transform = np.matmul(M_3x3, self.cumulative_transform)

    @staticmethod
    def apply_transform_to_bbox(bbox: List[float], M: np.ndarray) -> List[float]:
        """
        Transforms bounding box [x1, y1, x2, y2] using affine matrix M.
        """
        x1, y1, x2, y2 = bbox
        pts = np.array([
            [x1, y1, 1.0],
            [x2, y1, 1.0],
            [x1, y2, 1.0],
            [x2, y2, 1.0]
        ], dtype=np.float32).T  # 3x4

        transformed = np.matmul(M, pts)  # 2x4
        new_x1 = float(np.min(transformed[0, :]))
        new_y1 = float(np.min(transformed[1, :]))
        new_x2 = float(np.max(transformed[0, :]))
        new_y2 = float(np.max(transformed[1, :]))

        return [new_x1, new_y1, new_x2, new_y2]

    @staticmethod
    def apply_transform_to_point(point: Tuple[float, float], M: np.ndarray) -> Tuple[float, float]:
        """Transforms (x, y) point using affine matrix M."""
        x, y = point
        pt_vec = np.array([x, y, 1.0], dtype=np.float32)
        res = np.matmul(M, pt_vec)
        return float(res[0]), float(res[1])

    @staticmethod
    def warp_kalman_state(mean: np.ndarray, covariance: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warps Kalman Filter 8-dim state [x_c, y_c, a, h, vx, vy, va, vh] by affine matrix M.
        Ensures spatial motion prediction correctly shifts with camera ego-motion.
        """
        warped_mean = mean.copy()

        # Center position [x_c, y_c]
        cx, cy = mean[0], mean[1]
        pt = np.array([cx, cy, 1.0], dtype=np.float32)
        new_pos = np.matmul(M, pt)
        warped_mean[0] = new_pos[0]
        warped_mean[1] = new_pos[1]

        # Scale factor
        scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
        warped_mean[3] *= scale  # height changes with zoom
        warped_mean[7] *= scale  # vertical velocity changes with zoom

        # Velocity vectors [vx, vy] rotated by camera rotation
        R = M[:2, :2]
        vel = np.array([mean[4], mean[5]], dtype=np.float32)
        new_vel = np.matmul(R, vel)
        warped_mean[4] = new_vel[0]
        warped_mean[5] = new_vel[1]

        # Adjust covariance matrix using Jacobian transformation
        warped_cov = covariance.copy()
        # Scale positional uncertainties
        warped_cov[0, 0] *= (scale**2)
        warped_cov[1, 1] *= (scale**2)

        return warped_mean, warped_cov
