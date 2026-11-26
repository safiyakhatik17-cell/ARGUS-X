"""
Appearance-based Person Re-Identification (ReID)
Extracts deep appearance embeddings fused with spatial-color histograms
for invariant multi-object identity representation.
"""

from typing import List, Tuple, Optional, Union
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

from config import ReidConfig

# Ensure torch cache stays inside workspace
_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".cache", "torch"))
os.environ.setdefault("TORCH_HOME", _CACHE_DIR)


class LightweightReIDNet(nn.Module):
    """
    Lightweight feature extractor with Part-based Spatial Pooling.
    Outputs L2-normalized 512-dimensional appearance feature vectors.
    """
    def __init__(self, embedding_dim: int = 512, pretrained: bool = False):
        super().__init__()
        # Use MobileNetV3 backbone features
        weights = None
        if pretrained:
            try:
                weights = models.MobileNet_V3_Small_Weights.DEFAULT if hasattr(models, "MobileNet_V3_Small_Weights") else None
            except Exception:
                weights = None

        base = models.mobilenet_v3_small(weights=weights)
        self.features = base.features
        
        # Adaptive pooling to split feature maps into Upper body and Lower body parts
        self.gap_global = nn.AdaptiveAvgPool2d((1, 1))
        self.gap_upper = nn.AdaptiveAvgPool2d((2, 1)) # splits vertically
        
        in_channels = 576  # MobileNetV3-small output channel count
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Initialize linear layers
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.features(x) # [B, 576, H', W']
        
        # Global pooling: [B, 576]
        g_feat = self.gap_global(feat_map).flatten(1)
        
        # Upper and lower body pooling
        part_feat = self.gap_upper(feat_map) # [B, 576, 2, 1]
        upper = part_feat[:, :, 0, 0] # [B, 576]
        lower = part_feat[:, :, 1, 0] # [B, 576]
        part_combined = 0.5 * (upper + lower)
        
        concat = torch.cat([g_feat, part_combined], dim=1) # [B, 1152]
        emb = self.fc(concat) # [B, 512]
        
        # L2 normalize
        emb = F.normalize(emb, p=2, dim=1)
        return emb


class ReIDExtractor:
    """
    Extracts hybrid appearance embeddings (Deep CNN + Part-based HSV color histogram)
    to uniquely identify persons across frames and through occlusions.
    """

    def __init__(self, config: Optional[ReidConfig] = None):
        self.config = config or ReidConfig()
        
        # Determine device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Initialize network safely without relying on external web weights
        try:
            self.model = LightweightReIDNet(embedding_dim=self.config.embedding_dim, pretrained=False)
        except Exception:
            self.model = LightweightReIDNet(embedding_dim=self.config.embedding_dim, pretrained=False)
            
        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing pipeline
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.config.input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_crop_feature(self, crop: np.ndarray) -> np.ndarray:
        """
        Extracts appearance embedding from a single BGR image crop.
        Returns 1D numpy array of shape (512,).
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return np.zeros(self.config.embedding_dim, dtype=np.float32)

        # Deep embedding
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb_crop).unsqueeze(0).to(self.device)
        deep_emb = self.model(tensor).cpu().numpy().squeeze(0) # (512,)

        if not self.config.use_color_hist:
            return deep_emb

        # Color histogram signature (Upper body & Lower body HSV)
        color_emb = self._extract_part_color_hist(crop, bins=(8, 4, 4)) # 128-dim
        
        # Project and fuse
        w_col = self.config.color_hist_weight
        if len(color_emb) < self.config.embedding_dim:
            padded_col = np.zeros(self.config.embedding_dim, dtype=np.float32)
            padded_col[:len(color_emb)] = color_emb
        else:
            padded_col = color_emb[:self.config.embedding_dim]

        fused = (1.0 - w_col) * deep_emb + w_col * padded_col
        norm = np.linalg.norm(fused)
        if norm > 1e-6:
            fused = fused / norm

        return fused.astype(np.float32)

    @torch.no_grad()
    def extract_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extracts appearance embeddings for a list of crops.
        Returns 2D numpy array of shape (N, 512).
        """
        if not crops:
            return np.empty((0, self.config.embedding_dim), dtype=np.float32)

        features = []
        valid_tensors = []
        valid_indices = []

        for idx, crop in enumerate(crops):
            if crop is not None and crop.size > 0 and crop.shape[0] >= 4 and crop.shape[1] >= 4:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = self.transform(rgb_crop)
                valid_tensors.append(tensor)
                valid_indices.append(idx)

        # Batch forward pass for deep model
        deep_features = np.zeros((len(crops), self.config.embedding_dim), dtype=np.float32)
        if valid_tensors:
            batch_tensor = torch.stack(valid_tensors).to(self.device)
            deep_out = self.model(batch_tensor).cpu().numpy()
            for i, orig_idx in enumerate(valid_indices):
                deep_features[orig_idx] = deep_out[i]

        if not self.config.use_color_hist:
            return deep_features

        # Fuse color histograms
        fused_features = np.zeros_like(deep_features)
        w_col = self.config.color_hist_weight

        for idx, crop in enumerate(crops):
            if idx in valid_indices:
                col = self._extract_part_color_hist(crop, bins=(8, 4, 4))
                padded_col = np.zeros(self.config.embedding_dim, dtype=np.float32)
                padded_col[:min(len(col), self.config.embedding_dim)] = col[:self.config.embedding_dim]
                
                fused = (1.0 - w_col) * deep_features[idx] + w_col * padded_col
                norm = np.linalg.norm(fused)
                if norm > 1e-6:
                    fused = fused / norm
                fused_features[idx] = fused
            else:
                fused_features[idx] = np.zeros(self.config.embedding_dim, dtype=np.float32)

        return fused_features

    def _extract_part_color_hist(self, crop: np.ndarray, bins: Tuple[int, int, int] = (8, 4, 4)) -> np.ndarray:
        """
        Extracts HSV color histogram split into Upper Body and Lower Body halves.
        Provides strong invariance to pose flips and rotations.
        """
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        mid_h = h // 2

        # Upper half (torso/shirt)
        upper_hsv = hsv[:mid_h, :]
        hist_upper = cv2.calcHist([upper_hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist_upper, hist_upper)

        # Lower half (legs/pants)
        lower_hsv = hsv[mid_h:, :]
        hist_lower = cv2.calcHist([lower_hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist_lower, hist_lower)

        hist_combined = np.concatenate([hist_upper.flatten(), hist_lower.flatten()])
        norm = np.linalg.norm(hist_combined)
        if norm > 1e-6:
            hist_combined = hist_combined / norm

        return hist_combined.astype(np.float32)

    @staticmethod
    def compute_cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Computes cosine similarity between two 1D feature vectors."""
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        dot = np.dot(feat1, feat2) / (norm1 * norm2)
        return float(np.clip(dot, -1.0, 1.0))

    @staticmethod
    def compute_similarity_matrix(feats_a: np.ndarray, feats_b: np.ndarray) -> np.ndarray:
        """
        Computes pairwise cosine similarity matrix between feats_a (N, D) and feats_b (M, D).
        Returns matrix S of shape (N, M) with values in [-1, 1].
        """
        if feats_a.shape[0] == 0 or feats_b.shape[0] == 0:
            return np.empty((feats_a.shape[0], feats_b.shape[0]), dtype=np.float32)

        norm_a = np.linalg.norm(feats_a, axis=1, keepdims=True) + 1e-8
        norm_b = np.linalg.norm(feats_b, axis=1, keepdims=True) + 1e-8

        normed_a = feats_a / norm_a
        normed_b = feats_b / norm_b

        sim_matrix = np.matmul(normed_a, normed_b.T)
        return np.clip(sim_matrix, -1.0, 1.0)
