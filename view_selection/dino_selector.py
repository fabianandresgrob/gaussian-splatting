"""
DINO feature-based view selection strategy (improved implementation).

This selector uses DINO embeddings (DINOv2 or DINOv3) to sample views that 
maximize feature diversity, ensuring the training process sees semantically 
different views.

Supports:
    - DINOv2: dinov2_vitb14 (768 dim), dinov2_vitl14 (1024 dim), etc.
    - DINOv3: dinov3-vitl16 (1024 dim), dinov3-vitb16 (768 dim), etc.

Usage:
    1. First extract features for your scene:
       python tools/extract_dinov3_features.py --data_root ~/data/scenes/data --scene_id <scene_id>
    
    2. Use the selector in training:
       config = {
           'embeddings_path': '<scene_path>/dslr/dino_features/features.pt',
           'temperature': 1.0,
           'diversity_mode': 'distance_to_selected',
       }
       selector = build_selector('dino', config=config)
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional, Any
from scipy.special import softmax
from .selector import ViewSelector


class DINOSelector(ViewSelector):
    """
    Selector that prioritizes views based on DINO feature diversity.

    Uses precomputed DINO embeddings (v2 or v3) to ensure training samples
    semantically diverse views of the scene.

    Config parameters:
        - embeddings_path (str): Path to precomputed embeddings (.pt file)
        - temperature (float): Softmax temperature. Default: 1.0
          Higher values = more uniform, lower values = more peaked on diverse views
        - diversity_mode (str): How to compute diversity scores.
            - 'distance_to_centroid': Prioritize views far from feature centroid
            - 'distance_to_selected': Prioritize views far from recently selected (dynamic)
            - 'average_distance': Score by average distance to other views
            Default: 'distance_to_selected'
        - recency_window (int): Number of recent selections to consider 
          (for 'distance_to_selected' mode). Default: 50
        - normalize_embeddings (bool): L2 normalize embeddings. Default: True

    Precomputed embeddings format (from tools/extract_dinov3_features.py):
        {
            'DSC00001': tensor([...]),  # shape: (embedding_dim,)
            'DSC00002': tensor([...]),
            ...
            '_model': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
            '_embedding_dim': 1024,
        }
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        log_dir: Optional[str] = None,
        verbose: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize the DINO selector.

        Args:
            config: Configuration dictionary
            log_dir: Directory to save selection logs
            verbose: If True, print detailed information
            seed: Random seed for reproducibility
        """
        super().__init__(config, log_dir, verbose, seed)

        # embeddings_path options:
        # - explicit path to .pt
        # - "auto" to infer from camera.image_path at initialize()
        # - None to infer (best-effort) or fall back depending on flags
        self.embeddings_path = self.config.get('embeddings_path', None)
        self.temperature = self.config.get('temperature', 1.0)
        self.diversity_mode = self.config.get('diversity_mode', 'distance_to_selected')
        # Recency window: use large default to properly down-weight duplicates/similar views
        self.recency_window = self.config.get('recency_window', 69)
        self.normalize_embeddings = self.config.get('normalize_embeddings', True)
        
        # Track cumulative selection counts per camera for long-term diversity
        # NOTE: Disabled by default - this tracks by camera uid, not by content similarity,
        # so duplicates with different uids would be tracked separately. The recency-based
        # approach using feature similarity is the fair mechanism for diversity.
        self.selection_counts = None  # Will be initialized in initialize()
        self.use_cumulative_penalty = self.config.get('use_cumulative_penalty', False)

        # Behavior controls
        # If True, raise if embeddings cannot be located/loaded.
        # If False, fall back to uniform sampling (but log loudly).
        self.require_embeddings = bool(self.config.get('require_embeddings', False))

        # Will be populated during initialize()
        self.embeddings = {}  # Dict mapping image_name to embedding
        self.embedding_matrix = None  # (N, D) matrix for fast computation
        self.cam_idx_to_row = {}  # Map camera index to embedding matrix row
        self.recent_selections = []  # List of recently selected camera indices
        self.fixed_scores = None  # For static modes (centroid, average_distance)
        self.selection_counts = None  # Cumulative selection counts per camera

        self.logger.info(f"DINO Selector config: embeddings_path={self.embeddings_path}, "
                        f"temperature={self.temperature}, mode={self.diversity_mode}")

    def initialize(self, all_cameras: List) -> None:
        """
        Initialize with camera list and load DINO embeddings.

        Args:
            all_cameras: List of all available Camera objects
        """
        super().initialize(all_cameras)

        # Treat "auto" as a sentinel, not a literal filesystem path
        if self.embeddings_path == "auto":
            self.embeddings_path = None

        # Prefer resolving from source_path (passed from training script)
        source_path = self.config.get("source_path", None)
        if not self.embeddings_path and source_path:
            inferred = self._infer_embeddings_path_from_source_path(source_path)
            if inferred:
                self.embeddings_path = inferred
                self.logger.info(f"Inferred embeddings_path from source_path: {self.embeddings_path}")
            else:
                self.logger.warning(
                    f"Could not find DINO embeddings under source_path={source_path}. "
                    "Expected <source_path>/dino_features/features.pt."
                )

        # If embeddings_path is "auto" or not provided, try to infer it from the dataset layout
        # For ScanNet++/nerfstudio-style layouts, camera.image_path typically looks like:
        #   <scene_root>/resized_undistorted_images/<filename>.JPG
        # and embeddings are stored at:
        #   <scene_root>/dino_features/features.pt
        if not self.embeddings_path and self.all_cameras:
            inferred = self._infer_embeddings_path_from_cameras(self.all_cameras)
            if inferred:
                self.embeddings_path = inferred
                self.logger.info(f"Inferred embeddings_path: {self.embeddings_path}")
            else:
                self.logger.warning(
                    "Could not infer embeddings_path from camera image paths. "
                    "Expected <scene_root>/dino_features/features.pt."
                )

        # Load embeddings if path provided
        if self.embeddings_path:
            try:
                self._load_embeddings()
            except Exception as e:
                if self.require_embeddings:
                    raise
                self.logger.warning(f"Failed to load DINO embeddings ({self.embeddings_path}): {e}. Falling back to uniform sampling.")
        else:
            if self.require_embeddings:
                raise FileNotFoundError(
                    "DINOSelector requires embeddings, but no embeddings_path was provided and inference failed. "
                    "Set embeddings_path explicitly or use embeddings_path='auto' with a standard dataset layout."
                )

            self.logger.warning("No embeddings_path available. Falling back to uniform sampling.")
            self.logger.info("To use DINO features:")
            self.logger.info("  - Pass embeddings_path='<scene_root>/dino_features/features.pt'")
            self.logger.info("  - Or pass embeddings_path='auto' (recommended) and keep the standard folder layout")

        # Precompute static scores if using static mode
        if self.embedding_matrix is not None and self.diversity_mode in ['distance_to_centroid', 'average_distance']:
            self._compute_fixed_scores()
        
        # Initialize cumulative selection counts
        self.selection_counts = np.zeros(len(self.all_cameras))

        self.initialized = True

    def _infer_embeddings_path_from_cameras(self, cameras: List) -> Optional[str]:
        """Infer embeddings path from camera image paths if possible."""
        # Only attempt inference if camera objects expose image_path.
        cam0 = cameras[0]
        image_path = getattr(cam0, "image_path", None)
        if not image_path:
            return None

        # scene_root = <...>/dslr (or equivalent)
        # image_path = <scene_root>/resized_undistorted_images/<file>
        scene_root = os.path.dirname(os.path.dirname(image_path))
        candidates = [
            os.path.join(scene_root, "dino_features", "features.pt"),
            os.path.join(scene_root, "dino_features", "embeddings.pt"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    def _infer_embeddings_path_from_source_path(self, source_path: str) -> Optional[str]:
        """Infer embeddings path from dataset source path (recommended)."""
        candidates = [
            os.path.join(source_path, "dino_features", "features.pt"),
            os.path.join(source_path, "dino_features", "embeddings.pt"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    def _load_embeddings(self) -> None:
        """Load precomputed DINO embeddings from file."""
        if not os.path.exists(self.embeddings_path):
            raise FileNotFoundError(
                f"DINO embeddings not found: {self.embeddings_path}\n"
                f"Run tools/extract_dinov3_features.py first to precompute embeddings."
            )

        self.logger.info(f"Loading embeddings from {self.embeddings_path}")

        self.embeddings = torch.load(self.embeddings_path, map_location='cpu')

        # Extract metadata
        model_name = self.embeddings.pop('_model', 'unknown')
        embedding_dim = self.embeddings.pop('_embedding_dim', None)

        if not self.embeddings:
            raise ValueError(f"Empty embeddings file: {self.embeddings_path}")

        # Get embedding dimension from first entry if not in metadata
        if embedding_dim is None:
            first_embedding = next(iter(self.embeddings.values()))
            embedding_dim = first_embedding.shape[0]

        self.logger.info(f"Model: {model_name}, embedding dim: {embedding_dim}")

        # Build embedding matrix aligned with camera order
        embeddings_list: List[np.ndarray] = []
        missing_count = 0

        for cam_idx, cam in enumerate(self.all_cameras):
            image_name = cam.image_name
            
            # Try exact match first
            if image_name in self.embeddings:
                emb = self.embeddings[image_name]
            else:
                # Try without extension
                name_no_ext = os.path.splitext(image_name)[0]
                if name_no_ext in self.embeddings:
                    emb = self.embeddings[name_no_ext]
                else:
                    # Try fuzzy matching
                    emb = self._fuzzy_match_embedding(image_name, embedding_dim)
                    if emb is None:
                        emb = torch.zeros(embedding_dim)
                        missing_count += 1

            embeddings_list.append(emb.numpy() if isinstance(emb, torch.Tensor) else emb)
            self.cam_idx_to_row[cam_idx] = len(embeddings_list) - 1

        self.embedding_matrix = np.stack(embeddings_list)

        # Optionally normalize embeddings
        if self.normalize_embeddings:
            norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
            norms = np.where(norms > 1e-8, norms, 1.0)  # Avoid division by zero
            self.embedding_matrix = self.embedding_matrix / norms

        self.logger.info(f"Loaded embeddings for {len(self.all_cameras)} cameras, "
                        f"shape={self.embedding_matrix.shape}")
        if missing_count > 0:
            self.logger.warning(f"{missing_count} cameras missing embeddings (using zero vectors)")

    def _fuzzy_match_embedding(self, image_name: str, embedding_dim: int) -> Optional[torch.Tensor]:
        """Try to find embedding with fuzzy matching."""
        # Try various name formats
        for key in self.embeddings.keys():
            if image_name in key or key in image_name:
                return self.embeddings[key]
            # Handle DSC vs DSC_ prefixes
            if image_name.replace('DSC', 'DSC_') == key or key.replace('DSC', 'DSC_') == image_name:
                return self.embeddings[key]
        return None

    def _compute_fixed_scores(self) -> None:
        """Compute fixed diversity scores for static modes."""
        n_cameras = len(self.all_cameras)
        
        if self.diversity_mode == 'distance_to_centroid':
            # Score = distance from centroid (farther = more unique)
            centroid = np.mean(self.embedding_matrix, axis=0, keepdims=True)
            distances = np.linalg.norm(self.embedding_matrix - centroid, axis=1)
            self.fixed_scores = distances
            
        elif self.diversity_mode == 'average_distance':
            # Score = average distance to other views (higher = more unique)
            # Use cosine distance for efficiency
            similarity_matrix = self.embedding_matrix @ self.embedding_matrix.T
            # Convert to distance (1 - similarity for normalized vectors)
            distance_matrix = 1 - similarity_matrix
            # Average distance excluding self (diagonal)
            np.fill_diagonal(distance_matrix, 0)
            self.fixed_scores = distance_matrix.sum(axis=1) / (n_cameras - 1)

        self.logger.info(f"Computed fixed scores: min={self.fixed_scores.min():.4f}, "
                        f"max={self.fixed_scores.max():.4f}, "
                        f"mean={self.fixed_scores.mean():.4f}")

    def compute_probabilities(self, gaussians, iteration: int) -> Dict[int, float]:
        """
        Compute sampling probabilities based on DINO feature diversity.

        Args:
            gaussians: Current Gaussian model (unused)
            iteration: Current training iteration

        Returns:
            Dictionary mapping camera uid to sampling probability
        """
        n_cameras = len(self.all_cameras)

        # Fallback to uniform if no embeddings loaded
        if self.embedding_matrix is None:
            uniform_prob = 1.0 / n_cameras
            return {cam.uid: uniform_prob for cam in self.all_cameras}

        # Compute scores based on mode
        if self.diversity_mode == 'distance_to_selected':
            scores = self._compute_distance_to_selected_scores()
        elif self.diversity_mode in ['distance_to_centroid', 'average_distance']:
            scores = self.fixed_scores
        else:
            self.logger.warning(f"Unknown diversity_mode: {self.diversity_mode}, using uniform")
            scores = np.ones(n_cameras)

        # Convert to probabilities using softmax with temperature
        probabilities = softmax(scores / self.temperature)

        return {
            cam.uid: float(probabilities[i])
            for i, cam in enumerate(self.all_cameras)
        }

    def _compute_distance_to_selected_scores(self) -> np.ndarray:
        """
        Compute scores based on distance to recently selected views.

        Views far from recent selections get higher scores.
        Additionally applies cumulative penalty to down-weight frequently selected views.
        """
        n_cameras = len(self.all_cameras)

        if not self.recent_selections:
            # No history yet, fall back to distance from centroid
            centroid = np.mean(self.embedding_matrix, axis=0, keepdims=True)
            base_scores = np.linalg.norm(self.embedding_matrix - centroid, axis=1)
        else:
            # Get embeddings of recently selected cameras
            recent_indices = self.recent_selections[-self.recency_window:]
            recent_embeddings = self.embedding_matrix[recent_indices]

            # Compute similarity to recent selections (using dot product for normalized vectors)
            # Shape: (n_cameras, n_recent)
            similarities = self.embedding_matrix @ recent_embeddings.T
            
            # Score = 1 - max_similarity (furthest from any recent selection)
            max_similarities = similarities.max(axis=1)
            base_scores = 1.0 - max_similarities
        
        # Apply cumulative penalty: cameras selected many times get down-weighted
        # This ensures that even after recency window, frequently selected views
        # (like duplicates) accumulate penalty over time
        if self.use_cumulative_penalty and self.selection_counts is not None:
            # Penalty factor: 1 / (1 + alpha * count)
            # Higher counts -> lower multiplier -> lower final score
            alpha = 0.1  # Tunable: higher = stronger penalty for repeated selection
            penalty = 1.0 / (1.0 + alpha * self.selection_counts)
            scores = base_scores * penalty
        else:
            scores = base_scores

        return scores

    def log_selection(self, cam: Any, score: float, iteration: int) -> None:
        """Record selection for diversity tracking."""
        super().log_selection(cam, score, iteration)

        # Track camera index for distance computation
        for i, c in enumerate(self.all_cameras):
            if c.uid == cam.uid:
                self.recent_selections.append(i)
                # Update cumulative selection count
                if self.selection_counts is not None:
                    self.selection_counts[i] += 1
                break

        # Keep only recent selections to limit memory
        if len(self.recent_selections) > self.recency_window * 2:
            self.recent_selections = self.recent_selections[-self.recency_window:]

    def get_diversity_statistics(self) -> Dict[str, Any]:
        """Get statistics about feature diversity."""
        stats = {
            'diversity_mode': self.diversity_mode,
            'temperature': self.temperature,
            'recency_window': self.recency_window,
            'n_recent_selections': len(self.recent_selections),
        }
        
        if self.embedding_matrix is not None:
            # Compute pairwise similarities
            sim_matrix = self.embedding_matrix @ self.embedding_matrix.T
            np.fill_diagonal(sim_matrix, 0)
            
            stats['embedding_stats'] = {
                'n_embeddings': len(self.embedding_matrix),
                'embedding_dim': self.embedding_matrix.shape[1],
                'mean_pairwise_similarity': float(sim_matrix.mean()),
                'max_pairwise_similarity': float(sim_matrix.max()),
                'min_pairwise_similarity': float(sim_matrix.min()),
            }
        
        return stats
