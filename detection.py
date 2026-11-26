"""
YOLO Object Detection Module
Runs state-of-the-art YOLO detection with multi-class filtering,
confidence thresholding, and crop extraction for appearance ReID.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import cv2
from ultralytics import YOLO
import torch

from config import DetectorConfig


@dataclass
class Detection:
    """Standardized representation of a single object detection."""
    bbox: List[float]       # [x1, y1, x2, y2]
    confidence: float       # Detection confidence score [0.0, 1.0]
    class_id: int           # COCO class index (0 for person)
    class_name: str         # Human-readable class name
    crop: Optional[np.ndarray] = None # Extracted image patch


class YOLODetector:
    """
    Wraps Ultralytics YOLO with automatic device selection,
    class filtering, and safe image crop extraction.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        
        # Determine device
        if self.config.device == "auto":
            device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device_str = self.config.device

        # Load YOLO model
        self.model = YOLO(self.config.model_name)
        self.device_str = device_str

    def detect(
        self,
        frame: np.ndarray,
        classes: Optional[List[int]] = None,
        conf_threshold: Optional[float] = None
    ) -> List[Detection]:
        """
        Runs object detection on a single frame.
        
        Args:
            frame: BGR numpy image array.
            classes: List of class IDs to filter (defaults to config.target_classes, [0] for Person).
            conf_threshold: Custom confidence threshold.
            
        Returns:
            List of Detection objects.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        conf = conf_threshold if conf_threshold is not None else self.config.conf_threshold
        target_classes = classes if classes is not None else self.config.target_classes

        # Run inference
        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=self.config.nms_iou_threshold,
            classes=target_classes,
            device=self.device_str,
            verbose=False
        )

        detections: List[Detection] = []
        if not results:
            return detections

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return detections

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        names = res.names

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            # Clamp to image boundaries
            x1 = max(0.0, min(float(w - 1), float(x1)))
            y1 = max(0.0, min(float(h - 1), float(y1)))
            x2 = max(x1 + 1.0, min(float(w), float(x2)))
            y2 = max(y1 + 1.0, min(float(h), float(y2)))

            cid = int(cls_ids[i])
            cname = names.get(cid, f"class_{cid}")
            cconf = float(confs[i])

            # Extract crop
            ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            crop = frame[iy1:iy2, ix1:ix2].copy() if (iy2 > iy1 and ix2 > ix1) else None

            detections.append(Detection(
                bbox=[x1, y1, x2, y2],
                confidence=cconf,
                class_id=cid,
                class_name=cname,
                crop=crop
            ))

        return detections

    @staticmethod
    def extract_crops(frame: np.ndarray, detections: List[Detection]) -> List[np.ndarray]:
        """Safely extracts image crops for a list of detections."""
        h, w = frame.shape[:2]
        crops = []
        for det in detections:
            if det.crop is not None and det.crop.size > 0:
                crops.append(det.crop)
            else:
                x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                if x2 > x1 and y2 > y1:
                    crops.append(frame[y1:y2, x1:x2].copy())
                else:
                    crops.append(np.zeros((10, 10, 3), dtype=np.uint8))
        return crops
