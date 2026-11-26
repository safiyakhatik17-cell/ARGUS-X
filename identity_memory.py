"""
Long-Horizon Identity Memory Bank
Manages appearance signatures, exponential moving average (EMA) prototypes,
occlusion states, and identity recovery across hundreds of frames.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np

from config import IdentityMemoryConfig
from reid import ReIDExtractor


@dataclass
class IdentitySignature:
    """Persistent identity profile stored in long-horizon memory."""
    track_id: int
    mean_feature: np.ndarray             # EMA prototype feature vector (512,)
    feature_gallery: List[np.ndarray] = field(default_factory=list) # Exemplar features
    last_bbox: List[float] = field(default_factory=list) # [x1, y1, x2, y2]
    last_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32)) # [vx, vy]
    last_frame_idx: int = 0
    first_frame_idx: int = 0
    total_detections: int = 0
    recovery_count: int = 0
    state: str = "active"                # "active", "occluded", "lost", "recovered"
    avatar: Optional[np.ndarray] = None  # Best crop of this identity
    avatar_quality: float = 0.0          # Confidence/size metric for best crop
    trajectory: List[Tuple[int, float, float]] = field(default_factory=list) # (frame_idx, cx, cy)


class IdentityMemory:
    """
    Long-Horizon Identity Memory Bank.
    Resolves appearance drift, handles severe occlusions, and recovers
    lost identities when objects re-emerge.
    """

    def __init__(self, config: Optional[IdentityMemoryConfig] = None):
        self.config = config or IdentityMemoryConfig()
        # Gallery of all known identities: track_id -> IdentitySignature
        self.identities: Dict[int, IdentitySignature] = {}
        
        # Telemetry statistics
        self.total_recoveries: int = 0
        self.recovery_events: List[Dict[str, Any]] = []

    def reset(self):
        """Reset memory bank for a new session."""
        self.identities.clear()
        self.total_recoveries = 0
        self.recovery_events.clear()

    def register_or_update(
        self,
        track_id: int,
        feature: np.ndarray,
        bbox: List[float],
        frame_idx: int,
        velocity: Optional[np.ndarray] = None,
        crop: Optional[np.ndarray] = None,
        confidence: float = 1.0
    ):
        """
        Updates an existing identity or registers a new identity in memory.
        Uses EMA prototype updates and diverse exemplar clustering.
        """
        if feature is None or len(feature) == 0:
            return

        # Normalize feature vector
        norm = np.linalg.norm(feature)
        if norm > 1e-6:
            feature = feature / norm

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        vel = velocity if velocity is not None else np.zeros(2, dtype=np.float32)

        if track_id not in self.identities:
            # Register new identity
            sig = IdentitySignature(
                track_id=track_id,
                mean_feature=feature.copy(),
                feature_gallery=[feature.copy()],
                last_bbox=list(bbox),
                last_velocity=vel.copy(),
                last_frame_idx=frame_idx,
                first_frame_idx=frame_idx,
                total_detections=1,
                recovery_count=0,
                state="active",
                trajectory=[(frame_idx, cx, cy)]
            )
            # Store avatar
            if crop is not None and crop.size > 0:
                sig.avatar = crop.copy()
                sig.avatar_quality = confidence * (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            self.identities[track_id] = sig
        else:
            # Update existing identity
            sig = self.identities[track_id]
            sig.state = "active"
            sig.last_bbox = list(bbox)
            sig.last_velocity = vel.copy()
            sig.last_frame_idx = frame_idx
            sig.total_detections += 1
            sig.trajectory.append((frame_idx, cx, cy))

            # EMA update of mean appearance feature
            alpha = self.config.ema_alpha
            updated_mean = alpha * sig.mean_feature + (1.0 - alpha) * feature
            mean_norm = np.linalg.norm(updated_mean)
            if mean_norm > 1e-6:
                sig.mean_feature = updated_mean / mean_norm

            # Exemplar gallery update with diversity check
            self._update_gallery(sig, feature)

            # Update avatar if current crop is higher resolution/quality
            if crop is not None and crop.size > 0:
                qual = confidence * (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if qual > sig.avatar_quality:
                    sig.avatar = crop.copy()
                    sig.avatar_quality = qual

    def _update_gallery(self, sig: IdentitySignature, new_feature: np.ndarray):
        """Maintains diverse exemplar gallery for appearance variance."""
        if len(sig.feature_gallery) < self.config.gallery_capacity_per_id:
            sig.feature_gallery.append(new_feature.copy())
            return

        # Check maximum similarity against existing exemplars
        sims = [float(np.dot(new_feature, ex)) for ex in sig.feature_gallery]
        min_sim_idx = int(np.argmin(sims))

        # If novel enough compared to closest exemplar, replace the most redundant one
        if sims[min_sim_idx] < 0.92:
            sig.feature_gallery[min_sim_idx] = new_feature.copy()

    def mark_occluded_or_lost(self, track_id: int, current_frame_idx: int):
        """Marks a track as occluded or long-horizon lost based on elapsed frames."""
        if track_id in self.identities:
            sig = self.identities[track_id]
            elapsed = current_frame_idx - sig.last_frame_idx
            if elapsed < 30:
                sig.state = "occluded"
            else:
                sig.state = "lost"

    def match_and_recover(
        self,
        candidate_feature: np.ndarray,
        candidate_bbox: List[float],
        current_frame_idx: int,
        active_ids: List[int],
        camera_transform_cumulative: Optional[np.ndarray] = None
    ) -> Tuple[Optional[int], float]:
        """
        Queries the memory bank to find if an unmatched detection matches any lost/occluded identity.
        
        Args:
            candidate_feature: Feature embedding of newly detected object.
            candidate_bbox: Bounding box of newly detected object.
            current_frame_idx: Current frame index.
            active_ids: List of track IDs currently active in tracker (must not steal active IDs).
            camera_transform_cumulative: Cumulative camera transform to compensate spatial prediction.
            
        Returns:
            (recovered_track_id, match_confidence) or (None, 0.0)
        """
        if candidate_feature is None or len(candidate_feature) == 0:
            return None, 0.0

        best_id: Optional[int] = None
        best_score: float = 0.0

        cand_norm = np.linalg.norm(candidate_feature)
        if cand_norm < 1e-6:
            return None, 0.0
        norm_cand_feature = candidate_feature / cand_norm

        cand_cx = (candidate_bbox[0] + candidate_bbox[2]) / 2.0
        cand_cy = (candidate_bbox[1] + candidate_bbox[3]) / 2.0
        cand_diag = np.sqrt((candidate_bbox[2] - candidate_bbox[0])**2 + (candidate_bbox[3] - candidate_bbox[1])**2) + 1e-4

        for tid, sig in self.identities.items():
            # Never recover an ID that is already active
            if tid in active_ids:
                continue

            elapsed_frames = current_frame_idx - sig.last_frame_idx
            # Check maximum history window
            if elapsed_frames <= 0 or elapsed_frames > self.config.max_history_frames:
                continue

            # 1. Appearance similarity against EMA mean and exemplar gallery
            mean_sim = float(np.dot(norm_cand_feature, sig.mean_feature))
            gallery_sims = [float(np.dot(norm_cand_feature, ex)) for ex in sig.feature_gallery]
            max_gallery_sim = max(gallery_sims) if gallery_sims else mean_sim
            reid_sim = 0.4 * mean_sim + 0.6 * max_gallery_sim

            if reid_sim < self.config.reid_recovery_threshold * 0.90:
                continue

            # 2. Spatial motion extrapolation (Kalman velocity prediction)
            pred_cx = (sig.last_bbox[0] + sig.last_bbox[2]) / 2.0 + sig.last_velocity[0] * min(elapsed_frames, 45)
            pred_cy = (sig.last_bbox[1] + sig.last_bbox[3]) / 2.0 + sig.last_velocity[1] * min(elapsed_frames, 45)

            dist = np.sqrt((cand_cx - pred_cx)**2 + (cand_cy - pred_cy)**2)
            normalized_dist = dist / (cand_diag * 3.0)
            spatial_score = float(np.exp(-normalized_dist))

            # 3. Temporal decay weighting
            # As time passes, rely more on appearance and less on spatial proximity
            temp_weight = float(np.exp(-elapsed_frames / self.config.temporal_decay_half_life))
            w_spatial = self.config.spatial_consistency_weight * temp_weight

            # Total score
            composite_score = (1.0 - w_spatial) * reid_sim + w_spatial * spatial_score

            if composite_score > self.config.reid_recovery_threshold and composite_score > best_score:
                best_score = composite_score
                best_id = tid

        if best_id is not None:
            # Successfully recovered!
            sig = self.identities[best_id]
            sig.recovery_count += 1
            sig.state = "recovered"
            self.total_recoveries += 1
            
            elapsed = current_frame_idx - sig.last_frame_idx
            self.recovery_events.append({
                "frame": current_frame_idx,
                "track_id": best_id,
                "occlusion_duration_frames": elapsed,
                "score": float(best_score),
                "reid_sim": float(best_score)
            })

            return best_id, best_score

        return None, 0.0

    def get_identity_summary(self) -> List[Dict[str, Any]]:
        """Returns structured summaries for all tracked identities."""
        summary = []
        for tid, sig in self.identities.items():
            summary.append({
                "track_id": tid,
                "state": sig.state,
                "total_detections": sig.total_detections,
                "first_frame": sig.first_frame_idx,
                "last_frame": sig.last_frame_idx,
                "recoveries": sig.recovery_count,
                "has_avatar": sig.avatar is not None,
                "trajectory_points": len(sig.trajectory)
            })
        return summary
