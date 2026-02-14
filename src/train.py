"""
Training module for user authentication models.

This module provides the Trainer class which handles data preparation
and model training for individual user authentication.
"""

import os
import time
import numpy as np

from .model import Model
from .data_extraction import collect_segment_data
from .params import GeneticConfig


class Trainer:
    """
    Handles training of user-specific authentication models.

    Attributes:
        model: The Keras model being trained
        train_tensor: Numpy array of training data
        train_labels: Numpy array of binary labels (0=imposter, 1=owner)
        weights: Class weights to handle imbalanced data
    """

    def __init__(self, params, owner, others):
        """
        Initialize trainer and prepare training data.

        Args:
            params: Parameters object with window_sz, hz, and db
            owner: User ID of the authentic user (positive class)
            others: List of user IDs for imposters (negative class)
        """
        
        train_segments = GeneticConfig.TRAIN_SEGMENTS
        self.owner = owner
        self.topo_path = f"./models_cache/{params.window_sz}/{owner}/topo.json"
        self.weights_path = f"./models_cache/{params.window_sz}/{owner}/model.weights.h5"

        # Load or create model
        print("==> Acquiring model...", end='', flush=True)
        self.model = Model.get(params, self.topo_path, self.weights_path)
        print("done")

        # Collect training data
        print("==> Acquiring raw data...", end='', flush=True)
        pos_data = collect_segment_data(params, owner, train_segments)
        neg_data = []
        for other_id in others:
            neg_data.extend(collect_segment_data(params, other_id, train_segments))
        print("done")

        # Build training tensor
        print("==> Formatting tensor...", end='', flush=True)
        self.train_tensor, self.train_labels = self._build_tensor(
            pos_data, neg_data, params.window_sz, params.hz
        )
        print("done")

        # Calculate class weights (handle imbalanced data)
        print("==> Generating weights...", end='', flush=True)
        self.weights = {
            0: 1.0,
            1: len(neg_data) / len(pos_data)
        }
        print("done")

    @staticmethod
    def _build_tensor(pos_data, neg_data, window_sz, hz):
        """
        Build training tensor and labels from positive/negative samples.

        Args:
            pos_data: List of positive (owner) samples
            neg_data: List of negative (imposter) samples
            window_sz: Window size in seconds
            hz: Sampling frequency

        Returns:
            Tuple of (tensor, labels) numpy arrays
        """
        n_samples = len(pos_data) + len(neg_data)
        tensor = np.zeros((n_samples, 2, 3, window_sz * hz))

        # Negative samples first (label 0)
        for i, sample in enumerate(neg_data):
            tensor[i] = sample

        # Positive samples after (label 1)
        offset = len(neg_data)
        for i, sample in enumerate(pos_data):
            tensor[offset + i] = sample

        # Labels: 0s for negatives, 1s for positives
        labels = np.array([0] * len(neg_data) + [1] * len(pos_data))

        return tensor, labels

    def train(self, params, log_dir=None):
        """
        Train the model on prepared data.

        Args:
            params: Parameters object with epochs setting
            log_dir: Optional directory to save training logs

        Returns:
            Keras History object with training metrics
        """
        print("Training...", end='', flush=True)
        start_time = time.time()

        history = self.model.fit(
            self.train_tensor,
            self.train_labels,
            batch_size=32,
            epochs=params.epochs,
            verbose=0,
            class_weight=self.weights,
            shuffle=True
        )

        elapsed = time.time() - start_time
        print("done")

        Model.save_model(self.model, self.topo_path, self.weights_path)
        print(f"Model trained in {elapsed:.2f}s")

        # Save training accuracy log if log_dir specified
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{self.owner}.log")
            accuracies = history.history.get('accuracy', history.history.get('acc', []))
            with open(log_path, 'w') as f:
                f.write(f"accuracy = {accuracies}\n")
            print(f"Training log saved to {log_path}")

        return history
