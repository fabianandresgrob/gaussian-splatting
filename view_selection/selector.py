"""
Base class for view selection strategies in 3D Gaussian Splatting training.

This module provides an abstract interface for implementing different view selection
strategies to replace random camera sampling during training.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Dict, List, Optional
import json
import os


class ViewSelector(ABC):
    """
    Abstract base class for view selection strategies.

    Subclasses should implement different strategies for selecting which camera
    viewpoint to render from at each training iteration.
    """

    def __init__(self, config: dict, log_dir: Optional[str] = None, verbose: bool = False):
        """
        Initialize the view selector.

        Args:
            config: Dictionary containing strategy-specific configuration
            log_dir: Directory to save selection logs (if None, logging is disabled)
            verbose: If True, print detailed selection information
        """
        self.config = config
        self.log_dir = log_dir
        self.verbose = verbose
        self.selection_history = []  # List of (iteration, cam_uid, score) tuples
        self.selection_counts = {}  # Dict mapping cam_uid to selection count
        self.initialized = False
        self.all_cameras = []

        # Create log directory if needed
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_file = os.path.join(self.log_dir, "selection_history.jsonl")
        else:
            self.log_file = None

    @abstractmethod
    def initialize(self, all_cameras: List) -> None:
        """
        Pre-compute anything needed before training starts.

        This is called once after the scene is loaded, before the training loop begins.
        Use this to compute static features, cluster cameras, etc.

        Args:
            all_cameras: List of all available Camera objects
        """
        self.all_cameras = all_cameras

    @abstractmethod
    def compute_probabilities(self, gaussians, iteration: int) -> Dict[int, float]:
        """
        Compute sampling probabilities for each camera.

        Args:
            gaussians: Current Gaussian model (can be used for adaptive strategies)
            iteration: Current training iteration

        Returns:
            Dictionary mapping camera uid to sampling probability.
            Probabilities should sum to 1.0.
        """
        pass

    def select_view(self, gaussians, iteration: int):
        """
        Select a camera to render from.

        This is the main method called during training. It uses compute_probabilities()
        to get weights and samples accordingly.

        Args:
            gaussians: Current Gaussian model
            iteration: Current training iteration

        Returns:
            Selected Camera object
        """
        if not self.initialized:
            raise RuntimeError("ViewSelector must be initialized before use. Call initialize() first.")

        # Compute probabilities
        probabilities = self.compute_probabilities(gaussians, iteration)

        # Extract uids and probabilities in consistent order
        uids = list(probabilities.keys())
        probs = np.array([probabilities[uid] for uid in uids])

        # Normalize probabilities (in case of numerical errors)
        probs = probs / probs.sum()

        # Sample camera based on probabilities
        selected_idx = np.random.choice(len(uids), p=probs)
        selected_uid = uids[selected_idx]

        # Find the camera object
        selected_cam = None
        for cam in self.all_cameras:
            if cam.uid == selected_uid:
                selected_cam = cam
                break

        if selected_cam is None:
            raise RuntimeError(f"Camera with uid {selected_uid} not found in all_cameras")

        # Log the selection
        self.log_selection(selected_cam, probabilities[selected_uid], iteration)

        return selected_cam

    def log_selection(self, cam, score: float, iteration: int) -> None:
        """
        Record a camera selection for analysis.

        Args:
            cam: Selected Camera object
            score: Probability/score that led to this selection
            iteration: Current training iteration
        """
        # Update selection count
        if cam.uid not in self.selection_counts:
            self.selection_counts[cam.uid] = 0
        self.selection_counts[cam.uid] += 1

        # Add to history
        self.selection_history.append({
            'iteration': iteration,
            'cam_uid': cam.uid,
            'cam_name': cam.image_name,
            'probability': float(score),
            'selection_count': self.selection_counts[cam.uid]
        })

        # Write to log file
        if self.log_file:
            with open(self.log_file, 'a') as f:
                json.dump(self.selection_history[-1], f)
                f.write('\n')

        # Verbose output
        if self.verbose and iteration % 100 == 0:
            print(f"[Iter {iteration}] Selected camera {cam.uid} ({cam.image_name}) "
                  f"with probability {score:.4f} (selected {self.selection_counts[cam.uid]} times)")

    def get_selection_statistics(self) -> Dict:
        """
        Get statistics about camera selection patterns.

        Returns:
            Dictionary with selection statistics
        """
        stats = {
            'total_selections': len(self.selection_history),
            'unique_cameras': len(self.selection_counts),
            'selection_counts': self.selection_counts.copy(),
            'most_selected': max(self.selection_counts.items(), key=lambda x: x[1]) if self.selection_counts else None,
            'least_selected': min(self.selection_counts.items(), key=lambda x: x[1]) if self.selection_counts else None,
        }
        return stats

    def save_statistics(self, filepath: str) -> None:
        """
        Save selection statistics to a JSON file.

        Args:
            filepath: Path to save statistics
        """
        stats = self.get_selection_statistics()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)

    def log_statistics(self, iteration: int) -> None:
        """
        Print selection statistics to console.

        Args:
            iteration: Current training iteration
        """
        stats = self.get_selection_statistics()
        print(f"\n[Iter {iteration}] Selection Statistics:")
        print(f"  Total selections: {stats['total_selections']}")
        print(f"  Unique cameras used: {stats['unique_cameras']}")
        if stats['most_selected']:
            print(f"  Most selected: Camera {stats['most_selected'][0]} "
                  f"({stats['most_selected'][1]} times)")
        if stats['least_selected']:
            print(f"  Least selected: Camera {stats['least_selected'][0]} "
                  f"({stats['least_selected'][1]} times)")
