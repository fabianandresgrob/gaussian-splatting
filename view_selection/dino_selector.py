"""
DINO feature-based view selection strategy.

This selector uses DINOv2 embeddings to sample views that maximize feature
diversity, ensuring the training process sees semantically different views.

STATUS: PLACEHOLDER - Implementation required
TODO: Implement feature extraction and diversity-based sampling

Required implementation steps:
1. Precompute DINO embeddings for all training images (separate script)
2. Load embeddings during initialization
3. Compute diversity scores based on embedding distances
4. Convert to sampling probabilities

Recommended approach:
- Use dinov2_vitb14 (good balance of quality and speed)
- Precompute embeddings once per scene and save as .pt file
- Load embeddings as {image_name: embedding_tensor} dict
- Compute pairwise distances or use clustering for diversity
"""

import os
import numpy as np
from typing import Dict, List, Optional, Any
from scipy.special import softmax
from .selector import ViewSelector


class DINOSelector(ViewSelector):
    """
    Selector that prioritizes views based on DINO feature diversity.

    Uses precomputed DINOv2 embeddings to ensure training samples
    semantically diverse views of the scene.

    Config parameters:
        - embeddings_path (str): Path to precomputed embeddings (.pt file)
        - temperature (float): Softmax temperature. Default: 1.0
        - diversity_mode (str): How to compute diversity.
            - 'distance_to_selected': Prioritize views far from recently selected
            - 'clustering': Use embedding clusters, sample from underrepresented
            Default: 'distance_to_selected'
        - recency_window (int): Number of recent selections to consider. Default: 50

    Precomputed embeddings format:
        torch.save({
            'image_name_1': embedding_tensor_1,  # shape: (768,) for vitb14
            'image_name_2': embedding_tensor_2,
            ...
        }, 'embeddings.pt')
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
            config: Configuration dictionary with optional keys:
                - embeddings_path (str): Path to precomputed embeddings
                - temperature (float): Softmax temperature. Default: 1.0
                - diversity_mode (str): Diversity computation mode. Default: 'distance_to_selected'
                - recency_window (int): Recent selection window. Default: 50
            log_dir: Directory to save selection logs
            verbose: If True, print detailed information
            seed: Random seed for reproducibility
        """
        super().__init__(config, log_dir, verbose, seed)

        self.embeddings_path = self.config.get('embeddings_path', None)
        self.temperature = self.config.get('temperature', 1.0)
        self.diversity_mode = self.config.get('diversity_mode', 'distance_to_selected')
        self.recency_window = self.config.get('recency_window', 50)

        # Will be populated during initialize()
        self.embeddings = {}  # Dict mapping image_name to embedding
        self.embedding_matrix = None  # (N, D) matrix for fast computation
        self.recent_selections = []  # List of recently selected camera indices

        if self.verbose:
            self.logger.debug(f"Configuration: embeddings_path={self.embeddings_path}, "
                            f"temperature={self.temperature}, mode={self.diversity_mode}, "
                            f"recency_window={self.recency_window}")

    def initialize(self, all_cameras: List) -> None:
        """
        Initialize with camera list and load DINO embeddings.

        Args:
            all_cameras: List of all available Camera objects
        """
        super().initialize(all_cameras)

        # Load embeddings if path provided
        if self.embeddings_path:
            self._load_embeddings()
        else:
            # PLACEHOLDER: Fall back to uniform sampling
            self.logger.warning("No embeddings_path provided! Falling back to uniform sampling.")
            self.logger.info("To use DINO features: ")
            self.logger.info("  1. Run: python extract_dino_features.py --scene_path <path>")
            self.logger.info("  2. Pass: embeddings_path=<output_path> in config")

        self.initialized = True

    def _load_embeddings(self) -> None:
        """Load precomputed DINO embeddings from file."""
        import torch

        if not os.path.exists(self.embeddings_path):
            raise FileNotFoundError(
                f"DINO embeddings not found: {self.embeddings_path}\n"
                f"Run extract_dino_features.py first to precompute embeddings."
            )

        if self.verbose:
            self.logger.debug(f"Loading embeddings from {self.embeddings_path}")

        self.embeddings = torch.load(self.embeddings_path, map_location='cpu')

        if not self.embeddings:
            raise ValueError(f"Empty embeddings file: {self.embeddings_path}")

        # Get embedding dimension from first entry
        first_embedding = next(iter(self.embeddings.values()))
        embedding_dim = first_embedding.shape[0]

        # Build embedding matrix aligned with camera order
        embeddings_list: List[np.ndarray] = []
        missing_count = 0

        for cam in self.all_cameras:
            image_name = cam.image_name

            if image_name in self.embeddings:
                embeddings_list.append(self.embeddings[image_name].numpy())
            else:
                # Try alternative key formats
                found = False
                for key in self.embeddings.keys():
                    if image_name in key or key in image_name:
                        embeddings_list.append(self.embeddings[key].numpy())
                        found = True
                        break

                if not found:
                    # Use zero embedding as fallback
                    embeddings_list.append(np.zeros(embedding_dim))
                    missing_count += 1

        self.embedding_matrix = np.stack(embeddings_list)

        if self.verbose:
            self.logger.debug(f"Loaded {len(self.embeddings)} embeddings, shape={self.embedding_matrix.shape}")
            if missing_count > 0:
                self.logger.warning(f"{missing_count} cameras missing embeddings")

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

        # Compute diversity scores based on mode
        if self.diversity_mode == 'distance_to_selected':
            scores = self._compute_distance_scores()
        elif self.diversity_mode == 'clustering':
            scores = self._compute_clustering_scores()
        else:
            scores = np.ones(n_cameras)

        # Convert to probabilities
        probabilities = softmax(scores / self.temperature)

        return {
            cam.uid: float(probabilities[i])
            for i, cam in enumerate(self.all_cameras)
        }

    def _compute_distance_scores(self) -> np.ndarray:
        """
        Compute scores based on distance to recently selected views.

        Views far from recent selections get higher scores.
        """
        n_cameras = len(self.all_cameras)

        if not self.recent_selections:
            # No history yet - use uniform
            return np.ones(n_cameras)

        # Get embeddings of recently selected cameras
        recent_indices = self.recent_selections[-self.recency_window:]
        recent_embeddings = self.embedding_matrix[recent_indices]

        # Compute distance from each camera to nearest recent selection
        scores = np.zeros(n_cameras)

        for i in range(n_cameras):
            # Cosine distance to all recent selections
            embedding = self.embedding_matrix[i]
            distances = 1 - np.dot(recent_embeddings, embedding) / (
                np.linalg.norm(recent_embeddings, axis=1) * np.linalg.norm(embedding) + 1e-8
            )
            # Score is minimum distance (furthest from all recent)
            scores[i] = np.min(distances)

        return scores

    def _compute_clustering_scores(self) -> np.ndarray:
        """
        Compute scores based on embedding clusters.

        Views from underrepresented clusters get higher scores.
        """
        # TODO: Implement clustering-based diversity
        # This would cluster embeddings and weight inversely by cluster selection count
        return np.ones(len(self.all_cameras))

    def log_selection(self, cam: Any, score: float, iteration: int) -> None:
        """Record selection for diversity tracking."""
        super().log_selection(cam, score, iteration)

        # Track camera index for distance computation
        cam_idx: Optional[int] = None
        for i, c in enumerate(self.all_cameras):
            if c.uid == cam.uid:
                cam_idx = i
                break

        if cam_idx is not None:
            self.recent_selections.append(cam_idx)

            # Keep only recent selections
            if len(self.recent_selections) > self.recency_window * 2:
                self.recent_selections = self.recent_selections[-self.recency_window:]
