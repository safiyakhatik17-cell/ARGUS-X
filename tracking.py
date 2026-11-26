"""
Hierarchical Multi-Object Tracker (Track the Untrackable)
Combines Camera Motion Compensation, Kalman Filtering, Two-Stage ByteTrack Association,
and Long-Horizon Identity Memory Recovery to eliminate ID switches.
"""

from dataclasses import dataclass, replace
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import TrackerConfig
from camera_motion import CameraMotionCompensator
from reid import ReIDExtractor
from identity_memory import IdentityMemory
from detection import Detection


class TrackState(Enum):
    New = 0
    Tracked = 1
    Occluded = 2
    LongLost = 3
    Removed = 4


@dataclass(frozen=True)
class TrackObservation:
    """Track state for one object in one source-video frame."""

    frame_idx: int
    track_id: int
    bbox: Tuple[float, float, float, float]
    score: float
    state: TrackState
    is_recovered: bool = False
    recovery_similarity: float = 0.0
    time_since_update: int = 0


class KalmanBoxTracker:
    """
    Constant Velocity Model Kalman Filter for bounding box tracking.
    State: [cx, cy, aspect_ratio, height, v_cx, v_cy, v_a, v_h]
    """
    count = 0

    def __init__(self, bbox: List[float]):
        # State vector [8, 1]
        self.mean = np.zeros(8, dtype=np.float32)
        self.covariance = np.eye(8, dtype=np.float32)

        # Measurement matrix [4, 8]
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        # State transition matrix [8, 8]
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Motion noise covariance
        self.Q = np.eye(8, dtype=np.float32)
        self.Q[0:4, 0:4] *= 1.0
        self.Q[4:8, 4:8] *= 0.01

        # Measurement noise covariance
        self.R = np.eye(4, dtype=np.float32) * 1.0

        # Initialize state with measurement
        meas = self._bbox_to_z(bbox)
        self.mean[:4] = meas
        self.covariance[4:8, 4:8] *= 10.0

    def predict(self, M_camera: Optional[np.ndarray] = None):
        """
        Predicts next state. If M_camera affine matrix is provided,
        warps the prior state to compensate for camera ego-motion.
        """
        if M_camera is not None and not np.allclose(M_camera, np.eye(2, 3)):
            self.mean, self.covariance = CameraMotionCompensator.warp_kalman_state(
                self.mean, self.covariance, M_camera
            )

        # Standard Kalman prediction
        self.mean = np.matmul(self.F, self.mean)
        self.covariance = np.matmul(np.matmul(self.F, self.covariance), self.F.T) + self.Q

    def update(self, bbox: List[float]):
        """Kalman update step with new bounding box measurement."""
        z = self._bbox_to_z(bbox)
        y = z - np.matmul(self.H, self.mean) # Residual

        # Innovation covariance
        S = np.matmul(np.matmul(self.H, self.covariance), self.H.T) + self.R
        K = np.matmul(np.matmul(self.covariance, self.H.T), np.linalg.inv(S)) # Kalman gain

        self.mean = self.mean + np.matmul(K, y)
        I_KH = np.eye(8, dtype=np.float32) - np.matmul(K, self.H)
        self.covariance = np.matmul(I_KH, self.covariance)

    def get_bbox(self) -> List[float]:
        """Converts internal [cx, cy, a, h] state back to [x1, y1, x2, y2]."""
        return self._z_to_bbox(self.mean[:4])

    @staticmethod
    def _bbox_to_z(bbox: List[float]) -> np.ndarray:
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        a = w / h
        return np.array([cx, cy, a, h], dtype=np.float32)

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> List[float]:
        cx, cy, a, h = z[0], z[1], z[2], z[3]
        w = a * h
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return [float(x1), float(y1), float(x2), float(y2)]


class Track:
    """Represents a single continuous tracked object entity."""

    def __init__(self, track_id: int, bbox: List[float], score: float, feature: Optional[np.ndarray] = None):
        self.track_id = track_id
        self.kf = KalmanBoxTracker(bbox)
        self.bbox = list(bbox)
        self.score = float(score)
        self.feature = feature.copy() if feature is not None else None
        
        self.state = TrackState.New
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        
        # Identity recovery indicators
        self.is_recovered = False
        self.recovery_similarity = 0.0
        self.total_recoveries = 0

        # Trajectory history: list of (cx, cy)
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.trajectory: List[Tuple[float, float]] = [(cx, cy)]

    def predict(self, M_camera: Optional[np.ndarray] = None):
        """Predicts position and updates trajectory."""
        self.kf.predict(M_camera)
        self.bbox = self.kf.get_bbox()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox: List[float], score: float, feature: Optional[np.ndarray] = None):
        """Updates track with matched detection."""
        self.kf.update(bbox)
        self.bbox = self.kf.get_bbox()
        self.score = float(score)
        self.hits += 1
        self.time_since_update = 0

        if self.state == TrackState.New and self.hits >= 2:
            self.state = TrackState.Tracked
        elif self.state in (TrackState.Occluded, TrackState.LongLost):
            self.state = TrackState.Tracked

        if feature is not None:
            self.feature = feature.copy()

        # Update trajectory
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        self.trajectory.append((cx, cy))
        if len(self.trajectory) > 100:
            self.trajectory.pop(0)

    @property
    def velocity(self) -> np.ndarray:
        """Returns [vx, vy] velocity from Kalman filter."""
        return np.array([self.kf.mean[4], self.kf.mean[5]], dtype=np.float32)


class LongHorizonTracker:
    """
    Comprehensive Multi-Object Tracker with:
    - Global Camera Motion Compensation (GMC)
    - Two-stage association (ByteTrack principle)
    - Deep ReID + Spatial appearance fusion
    - Long-Horizon Identity Memory recovery under severe occlusions
    """

    def __init__(
        self,
        tracker_config: Optional[TrackerConfig] = None,
        reid_extractor: Optional[ReIDExtractor] = None,
        camera_compensator: Optional[CameraMotionCompensator] = None,
        identity_memory: Optional[IdentityMemory] = None
    ):
        self.config = tracker_config or TrackerConfig()
        self.reid = reid_extractor or ReIDExtractor()
        self.gmc = camera_compensator or CameraMotionCompensator()
        self.memory = identity_memory or IdentityMemory()

        self.tracks: List[Track] = []
        self._next_id = 1
        self.frame_idx = 0

        # Global performance metrics
        self.total_tracks_created = 0
        self.total_recoveries = 0
        self.id_switches_prevented = 0

    def reset(self):
        """Resets tracker state."""
        self.tracks.clear()
        self._next_id = 1
        self.frame_idx = 0
        self.total_tracks_created = 0
        self.total_recoveries = 0
        self.id_switches_prevented = 0
        self.gmc.reset()
        self.memory.reset()

    def update(self, frame: np.ndarray, detections: List[Detection]) -> List[Track]:
        """
        Processes a video frame with detections and returns active tracked objects.
        """
        self.frame_idx += 1
        h, w = frame.shape[:2]

        # 1. Compute Camera Ego-Motion
        det_boxes = [d.bbox for d in detections]
        M_camera = self.gmc.compute_motion(frame, exclusion_boxes=det_boxes) if self.config.gmc_enabled else None

        # 2. Predict positions of existing tracks with camera motion compensation
        for track in self.tracks:
            track.predict(M_camera)
            # Reset recovery indicator for current frame
            track.is_recovered = False

        # 3. Partition detections into High-Confidence and Low-Confidence
        high_dets = [d for d in detections if d.confidence >= self.config.track_high_thresh]
        low_dets = [d for d in detections if self.config.track_low_thresh <= d.confidence < self.config.track_high_thresh]

        # Extract ReID appearance embeddings for high-confidence detections
        high_crops = [d.crop for d in high_dets]
        high_features = self.reid.extract_batch(high_crops)

        # Split tracks into confirmed and unconfirmed
        confirmed_tracks = [t for t in self.tracks if t.state == TrackState.Tracked]
        unconfirmed_tracks = [t for t in self.tracks if t.state in (TrackState.New, TrackState.Occluded)]

        # -------------------------------------------------------------
        # Stage 1: High-Confidence Detections vs Confirmed Tracks
        # Combined IoU + ReID Cosine Distance Matching
        # -------------------------------------------------------------
        matches_1, unmatched_tracks_1, unmatched_dets_1 = self._match_stage_1(
            confirmed_tracks, high_dets, high_features
        )

        for t_idx, d_idx in matches_1:
            track = confirmed_tracks[t_idx]
            det = high_dets[d_idx]
            feat = high_features[d_idx]
            track.update(det.bbox, det.confidence, feat)
            # Update Identity Memory bank
            self.memory.register_or_update(
                track.track_id, feat, track.bbox, self.frame_idx,
                velocity=track.velocity, crop=det.crop, confidence=det.confidence
            )

        # -------------------------------------------------------------
        # Stage 2: Low-Confidence Detections vs Remaining Active Tracks
        # Recovers temporarily blurred/occluded objects without appearance noise
        # -------------------------------------------------------------
        remaining_tracks_stage2 = [confirmed_tracks[i] for i in unmatched_tracks_1]
        matches_2, unmatched_tracks_2, _ = self._match_iou(
            remaining_tracks_stage2, low_dets, self.config.match_iou_thresh
        )

        for t_idx, d_idx in matches_2:
            track = remaining_tracks_stage2[t_idx]
            det = low_dets[d_idx]
            track.update(det.bbox, det.confidence, None)

        # Remaining unmatched tracks from Stage 1 & 2 enter Occluded status
        remaining_unmatched_confirmed = [remaining_tracks_stage2[i] for i in unmatched_tracks_2]
        for t in remaining_unmatched_confirmed:
            if t.state == TrackState.Tracked:
                t.state = TrackState.Occluded
                self.memory.mark_occluded_or_lost(t.track_id, self.frame_idx)

        # -------------------------------------------------------------
        # Stage 3: High-Confidence Unmatched vs Unconfirmed Tracks
        # -------------------------------------------------------------
        unmatched_high_dets = [high_dets[i] for i in unmatched_dets_1]
        unmatched_high_feats = [high_features[i] for i in unmatched_dets_1]

        matches_3, unmatched_unconfirmed, unmatched_dets_3 = self._match_iou(
            unconfirmed_tracks, unmatched_high_dets, self.config.unconfirmed_iou_thresh
        )

        for t_idx, d_idx in matches_3:
            track = unconfirmed_tracks[t_idx]
            det = unmatched_high_dets[d_idx]
            feat = unmatched_high_feats[d_idx]
            track.update(det.bbox, det.confidence, feat)
            self.memory.register_or_update(
                track.track_id, feat, track.bbox, self.frame_idx,
                velocity=track.velocity, crop=det.crop, confidence=det.confidence
            )

        # -------------------------------------------------------------
        # Stage 4: Long-Horizon Identity Memory Recovery
        # Queries memory bank for lost identities to prevent ID switches
        # -------------------------------------------------------------
        final_unmatched_dets = [unmatched_high_dets[i] for i in unmatched_dets_3]
        final_unmatched_feats = [unmatched_high_feats[i] for i in unmatched_dets_3]

        active_track_ids = [t.track_id for t in self.tracks if t.time_since_update == 0]

        new_tracks: List[Track] = []
        for det, feat in zip(final_unmatched_dets, final_unmatched_feats):
            if det.confidence < self.config.new_track_thresh:
                continue

            recovered_id: Optional[int] = None
            sim_score: float = 0.0

            if self.config.enable_identity_recovery:
                recovered_id, sim_score = self.memory.match_and_recover(
                    feat, det.bbox, self.frame_idx, active_track_ids,
                    camera_transform_cumulative=self.gmc.cumulative_transform
                )

            if recovered_id is not None:
                # Identity RECOVERED! Link back to original global identity
                self.total_recoveries += 1
                self.id_switches_prevented += 1
                active_track_ids.append(recovered_id)

                # Check if existing occluded track object is in self.tracks
                existing_track = next((t for t in self.tracks if t.track_id == recovered_id), None)
                if existing_track is not None:
                    existing_track.update(det.bbox, det.confidence, feat)
                    existing_track.state = TrackState.Tracked
                    existing_track.is_recovered = True
                    existing_track.recovery_similarity = float(sim_score)
                    existing_track.total_recoveries += 1
                    track_to_record = existing_track
                else:
                    track = Track(recovered_id, det.bbox, det.confidence, feat)
                    track.state = TrackState.Tracked
                    track.is_recovered = True
                    track.recovery_similarity = float(sim_score)
                    track.total_recoveries += 1
                    new_tracks.append(track)
                    track_to_record = track

                self.memory.register_or_update(
                    recovered_id, feat, det.bbox, self.frame_idx,
                    velocity=track_to_record.velocity, crop=det.crop, confidence=det.confidence
                )
            else:
                # Completely novel object: initialize brand new track ID
                new_id = self._next_id
                self._next_id += 1
                track = Track(new_id, det.bbox, det.confidence, feat)
                self.total_tracks_created += 1
                new_tracks.append(track)
                active_track_ids.append(new_id)

                self.memory.register_or_update(
                    new_id, feat, det.bbox, self.frame_idx,
                    velocity=track.velocity, crop=det.crop, confidence=det.confidence
                )

        # -------------------------------------------------------------
        # Clean up stale tracks
        # -------------------------------------------------------------
        self.tracks.extend(new_tracks)
        retained_tracks: List[Track] = []

        for t in self.tracks:
            if t.state == TrackState.Removed:
                continue

            if t.time_since_update > self.config.track_buffer:
                # Mark as LongLost in identity memory so it remains recoverable indefinitely
                t.state = TrackState.LongLost
                self.memory.mark_occluded_or_lost(t.track_id, self.frame_idx)
                # Remove from active tracker list
                continue

            retained_tracks.append(t)

        self.tracks = retained_tracks

        # Return tracks that are actively confirmed or currently occluded (with predicted bbox)
        return [t for t in self.tracks if t.state in (TrackState.Tracked, TrackState.Occluded)]

    def _match_stage_1(
        self,
        tracks: List[Track],
        dets: List[Detection],
        det_features: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Stage 1 Matching: Cost = (1 - w) * (1 - IoU) + w * (1 - CosSim)
        """
        if len(tracks) == 0 or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))

        # IoU matrix
        iou_mat = self._compute_iou_matrix([t.bbox for t in tracks], [d.bbox for d in dets])
        iou_dist = 1.0 - iou_mat

        # ReID Cosine Distance
        track_feats = np.array([t.feature for t in tracks if t.feature is not None])
        if len(track_feats) == len(tracks) and len(det_features) == len(dets):
            sim_mat = ReIDExtractor.compute_similarity_matrix(track_feats, det_features)
            reid_dist = 1.0 - np.clip(sim_mat, 0.0, 1.0)
        else:
            reid_dist = np.ones_like(iou_dist)

        w_reid = self.config.reid_distance_weight
        cost_matrix = (1.0 - w_reid) * iou_dist + w_reid * reid_dist

        # Gating: if IoU is 0 and ReID distance is high, forbid match
        cost_matrix[iou_mat < 0.10] += 2.0

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches: List[Tuple[int, int]] = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(dets)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < (1.0 - self.config.match_iou_thresh * (1.0 - w_reid)):
                matches.append((r, c))
                if r in unmatched_tracks:
                    unmatched_tracks.remove(r)
                if c in unmatched_dets:
                    unmatched_dets.remove(c)

        return matches, unmatched_tracks, unmatched_dets

    def _match_iou(
        self,
        tracks: List[Track],
        dets: List[Any],
        thresh: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Standard spatial IoU linear assignment."""
        if len(tracks) == 0 or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))

        det_boxes = [d.bbox for d in dets]
        iou_mat = self._compute_iou_matrix([t.bbox for t in tracks], det_boxes)
        cost_matrix = 1.0 - iou_mat

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches: List[Tuple[int, int]] = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(dets)))

        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= thresh:
                matches.append((r, c))
                if r in unmatched_tracks:
                    unmatched_tracks.remove(r)
                if c in unmatched_dets:
                    unmatched_dets.remove(c)

        return matches, unmatched_tracks, unmatched_dets

    @staticmethod
    def _compute_iou_matrix(boxes_a: List[List[float]], boxes_b: List[List[float]]) -> np.ndarray:
        """Computes pairwise Intersection over Union (IoU) matrix."""
        na = len(boxes_a)
        nb = len(boxes_b)
        if na == 0 or nb == 0:
            return np.empty((na, nb), dtype=np.float32)

        a = np.array(boxes_a, dtype=np.float32)
        b = np.array(boxes_b, dtype=np.float32)

        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        # Pairwise corners
        x1 = np.maximum(a[:, None, 0], b[None, :, 0])
        y1 = np.maximum(a[:, None, 1], b[None, :, 1])
        x2 = np.minimum(a[:, None, 2], b[None, :, 2])
        y2 = np.minimum(a[:, None, 3], b[None, :, 3])

        w = np.maximum(0.0, x2 - x1)
        h = np.maximum(0.0, y2 - y1)
        intersection = w * h

        union = area_a[:, None] + area_b[None, :] - intersection
        return np.where(union > 0, intersection / np.maximum(union, 1e-6), 0.0)


def merge_bidirectional_observations(
    forward_frames: List[List[TrackObservation]],
    backward_frames: List[List[TrackObservation]],
    min_match_iou: float = 0.30
) -> List[List[TrackObservation]]:
    """Reconcile reverse-pass IDs to forward-pass IDs and fill missing observations."""
    if len(forward_frames) != len(backward_frames):
        raise ValueError("Forward and backward results must contain the same frame count.")

    forward_ids = sorted({obs.track_id for frame in forward_frames for obs in frame})
    backward_ids = sorted({obs.track_id for frame in backward_frames for obs in frame})
    if not backward_ids:
        return [list(frame) for frame in forward_frames]

    forward_index = {track_id: index for index, track_id in enumerate(forward_ids)}
    backward_index = {track_id: index for index, track_id in enumerate(backward_ids)}
    overlap_scores = np.zeros((len(forward_ids), len(backward_ids)), dtype=np.float64)

    for forward_frame, backward_frame in zip(forward_frames, backward_frames):
        if not forward_frame or not backward_frame:
            continue
        iou = LongHorizonTracker._compute_iou_matrix(
            [list(obs.bbox) for obs in forward_frame],
            [list(obs.bbox) for obs in backward_frame]
        )
        rows, cols = linear_sum_assignment(1.0 - iou)
        for row, col in zip(rows, cols):
            if iou[row, col] >= min_match_iou:
                forward_id = forward_frame[row].track_id
                backward_id = backward_frame[col].track_id
                overlap_scores[forward_index[forward_id], backward_index[backward_id]] += iou[row, col]

    rows, cols = linear_sum_assignment(-overlap_scores)
    backward_to_forward = {
        backward_ids[col]: forward_ids[row]
        for row, col in zip(rows, cols)
        if overlap_scores[row, col] > 0.0
    }

    next_id = max(forward_ids, default=0) + 1
    for track_id in backward_ids:
        if track_id not in backward_to_forward:
            backward_to_forward[track_id] = next_id
            next_id += 1

    state_priority = {
        TrackState.Tracked: 3,
        TrackState.New: 2,
        TrackState.Occluded: 1,
        TrackState.LongLost: 0,
        TrackState.Removed: -1,
    }
    merged_frames: List[List[TrackObservation]] = []
    for forward_frame, backward_frame in zip(forward_frames, backward_frames):
        merged = {obs.track_id: obs for obs in forward_frame}
        for obs in backward_frame:
            canonical_id = backward_to_forward[obs.track_id]
            remapped = replace(obs, track_id=canonical_id)
            current = merged.get(canonical_id)
            if current is None or (
                state_priority[remapped.state], remapped.score
            ) > (
                state_priority[current.state], current.score
            ):
                merged[canonical_id] = remapped
        merged_frames.append([merged[track_id] for track_id in sorted(merged)])

    return merged_frames
