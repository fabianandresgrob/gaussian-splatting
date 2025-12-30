"""
Clustering-based view selection strategy.

This selector clusters cameras by pose (position + orientation) and samples
inversely proportional to cluster size, ensuring underrepresented viewpoints
are prioritized.

Supports both K-Means and DBSCAN clustering algorithms.
"""

import numpy as np
from typing import Dict, List
from scipy.special import softmax
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from .selector import ViewSelector


class ClusteringSelector(ViewSelector):
    """
    Selector that clusters cameras and prioritizes underrepresented clusters.

    Process:
    1. Cluster cameras by pose (position + viewing direction)
    2. Assign probabilities inversely proportional to cluster size
    3. Dynamically adjust based on per-cluster selection counts

    Supports K-Means (fixed cluster count) and DBSCAN (adaptive cluster count).
    """

    def __init__(self, config: dict = None, log_dir: str = None, verbose: bool = False, seed: int = None):
        """
        Initialize the clustering selector.

        Args:
            config: Configuration dictionary with optional keys:
                - clustering_method (str): 'kmeans' or 'dbscan'. Default: 'kmeans'
                - n_clusters (int): Number of clusters (K-Means only). Default: 10
                - eps (float): DBSCAN epsilon parameter. Default: 0.5
                - min_samples (int): DBSCAN min_samples parameter. Default: 3
                - temperature (float): Softmax temperature. Default: 1.0
                - use_orientation (bool): Include viewing direction in clustering. Default: True
                - update_frequency (int): How often to recompute probabilities (in iterations).
                  Default: 100 (recompute every 100 iterations)
            log_dir: Directory to save selection logs
            verbose: If True, print detailed information
            seed: Random seed for reproducibility
        """
        super().__init__(config or {}, log_dir, verbose, seed=seed)
        self.clustering_method = self.config.get('clustering_method', 'kmeans')
        self.n_clusters = self.config.get('n_clusters', 10)
        self.eps = self.config.get('eps', 0.5)
        self.min_samples = self.config.get('min_samples', 3)
        self.temperature = self.config.get('temperature', 1.0)
        self.use_orientation = self.config.get('use_orientation', True)
        self.update_frequency = self.config.get('update_frequency', 100)

        self.clusterer = None  # Will hold KMeans or DBSCAN instance
        self.camera_clusters = {}  # Maps camera uid to cluster id
        self.cluster_sizes = {}  # Maps cluster id to number of cameras
        self.cluster_selection_counts = {}  # Maps cluster id to selection count
        self.last_update_iteration = 0
        self.current_probabilities = None

    def initialize(self, all_cameras: List) -> None:
        """
        Cluster cameras by pose using K-Means or DBSCAN.

        Args:
            all_cameras: List of all available Camera objects
        """
        super().initialize(all_cameras)
        self.logger.info(f"Initializing with {len(all_cameras)} cameras")
        self.logger.info(f"Clustering method: {self.clustering_method}")
        if self.clustering_method == 'kmeans':
            self.logger.debug(f"Number of clusters: {self.n_clusters}")
        else:
            self.logger.debug(f"DBSCAN eps: {self.eps}, min_samples: {self.min_samples}")
        self.logger.debug(f"Use orientation: {self.use_orientation}")
        self.logger.debug(f"Update frequency: {self.update_frequency} iterations")

        # Extract features for clustering
        features = []
        for cam in all_cameras:
            # Camera position
            pos = cam.camera_center.cpu().numpy()
            feature = list(pos)

            if self.use_orientation:
                # Compute viewing direction from rotation matrix
                # The camera looks down the negative Z axis in camera space
                # Transform to world space using R^T (since R transforms world to camera)
                R = cam.R  # This is world-to-camera rotation
                view_dir = -R[2, :]  # Third row gives the forward direction
                feature.extend(view_dir)

            features.append(feature)

        features = np.array(features)

        # Standardize features before clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Perform clustering based on method
        if self.clustering_method == 'kmeans':
            # Adjust n_clusters if there are fewer cameras
            actual_n_clusters = min(self.n_clusters, len(all_cameras))
            if actual_n_clusters < self.n_clusters:
                self.logger.warning(f"Only {len(all_cameras)} cameras available, "
                      f"reducing clusters to {actual_n_clusters}")
                self.n_clusters = actual_n_clusters

            # Perform K-Means clustering
            self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
            cluster_labels = self.clusterer.fit_predict(features_scaled)

        elif self.clustering_method == 'dbscan':
            # Perform DBSCAN clustering
            self.clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            cluster_labels = self.clusterer.fit_predict(features_scaled)

            # Handle noise points (-1 label) by assigning them to a separate "outlier" cluster
            # Find the max cluster id (excluding -1)
            max_cluster = cluster_labels.max()
            outlier_cluster_id = max_cluster + 1 if max_cluster >= 0 else 0

            # Replace -1 with outlier cluster id
            cluster_labels_adjusted = cluster_labels.copy()
            cluster_labels_adjusted[cluster_labels == -1] = outlier_cluster_id
            cluster_labels = cluster_labels_adjusted

            # Determine actual number of clusters (including outlier cluster if present)
            self.n_clusters = len(np.unique(cluster_labels))

            if self.verbose:
                n_noise = np.sum(self.clusterer.labels_ == -1)
                n_regular_clusters = len(np.unique(self.clusterer.labels_[self.clusterer.labels_ != -1]))
                self.logger.debug(f"DBSCAN found {n_regular_clusters} clusters")
                if n_noise > 0:
                    self.logger.debug(f"{n_noise} noise points assigned to outlier cluster {outlier_cluster_id}")

        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}. "
                           f"Must be 'kmeans' or 'dbscan'")

        # Store cluster assignments
        self.camera_clusters = {cam.uid: int(cluster_labels[i]) for i, cam in enumerate(all_cameras)}

        # Count cameras per cluster
        unique_clusters = np.unique(cluster_labels)
        self.cluster_sizes = {}
        for cluster_id in unique_clusters:
            self.cluster_sizes[int(cluster_id)] = int(np.sum(cluster_labels == cluster_id))

        # Initialize selection counts
        self.cluster_selection_counts = {int(cluster_id): 0 for cluster_id in unique_clusters}

        if self.verbose:
            self.logger.debug(f"Final number of clusters: {self.n_clusters}")
            self.logger.debug(f"Cluster sizes: {self.cluster_sizes}")
            avg_size = np.mean(list(self.cluster_sizes.values()))
            self.logger.debug(f"Average cluster size: {avg_size:.2f}")

        # Compute initial probabilities
        self.current_probabilities = self._compute_probabilities_internal()

        self.initialized = True

    def _compute_probabilities_internal(self) -> Dict[int, float]:
        """
        Internal method to compute probabilities based on current cluster statistics.

        Returns:
            Dictionary mapping camera uid to probability
        """
        # Compute score for each cluster
        # Score is inversely proportional to: (cluster_size + selection_count)
        cluster_scores = {}
        for cluster_id in self.cluster_sizes.keys():
            base_size = self.cluster_sizes[cluster_id]
            selection_count = self.cluster_selection_counts[cluster_id]
            # Higher score for smaller, less-selected clusters
            cluster_scores[cluster_id] = 1.0 / (base_size + selection_count + 1e-8)

        # Assign each camera the score of its cluster
        camera_scores = {}
        for cam in self.all_cameras:
            cluster_id = self.camera_clusters[cam.uid]
            camera_scores[cam.uid] = cluster_scores[cluster_id]

        # Convert to probabilities using softmax
        uids = list(camera_scores.keys())
        scores = np.array([camera_scores[uid] for uid in uids])
        probabilities = softmax(scores / self.temperature)

        probabilities_dict = {
            uid: float(probabilities[i])
            for i, uid in enumerate(uids)
        }

        return probabilities_dict

    def compute_probabilities(self, gaussians, iteration: int) -> Dict[int, float]:
        """
        Compute probabilities based on cluster statistics.

        Recomputes every update_frequency iterations to adapt to selection patterns.

        Args:
            gaussians: Current Gaussian model (unused)
            iteration: Current training iteration

        Returns:
            Dictionary mapping camera uid to probability
        """
        # Check if we need to recompute probabilities
        if iteration - self.last_update_iteration >= self.update_frequency:
            self.current_probabilities = self._compute_probabilities_internal()
            self.last_update_iteration = iteration

            if self.verbose and iteration > 0:
                self.logger.debug(f"Updated probabilities at iteration {iteration}")
                self.logger.debug(f"Cluster selection counts: {self.cluster_selection_counts}")

        return self.current_probabilities

    def log_selection(self, cam, score: float, iteration: int) -> None:
        """
        Record selection and update cluster statistics.

        Args:
            cam: Selected Camera object
            score: Probability that led to this selection
            iteration: Current training iteration
        """
        # Call parent logging
        super().log_selection(cam, score, iteration)

        # Update cluster selection count
        cluster_id = self.camera_clusters[cam.uid]
        self.cluster_selection_counts[cluster_id] += 1

    def get_cluster_statistics(self) -> Dict:
        """
        Get statistics about cluster selection patterns.

        Returns:
            Dictionary with cluster statistics
        """
        stats = {
            'clustering_method': self.clustering_method,
            'n_clusters': self.n_clusters,
            'cluster_sizes': self.cluster_sizes,
            'cluster_selection_counts': self.cluster_selection_counts,
            'selections_per_camera_per_cluster': {
                cluster_id: self.cluster_selection_counts[cluster_id] / self.cluster_sizes[cluster_id]
                if self.cluster_sizes[cluster_id] > 0 else 0
                for cluster_id in self.cluster_sizes.keys()
            }
        }
        return stats

    def save_statistics(self, filepath: str) -> None:
        """
        Save both general and cluster-specific statistics.

        Args:
            filepath: Path to save statistics
        """
        import json
        stats = self.get_selection_statistics()
        cluster_stats = self.get_cluster_statistics()
        combined_stats = {**stats, **cluster_stats}
        with open(filepath, 'w') as f:
            json.dump(combined_stats, f, indent=2)
