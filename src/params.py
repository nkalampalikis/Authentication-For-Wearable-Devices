"""
Configuration parameters for the authentication system.

This module provides the Parameters class which holds all configuration
for data loading, model architecture, and training, as well as genetic
algorithm configuration.
"""

import os
from .db import DB
from .signal_filter import SignalType


# =============================================================================
# Genetic Algorithm Configuration
# =============================================================================

def _build_traits_dict():
    """Build the dictionary of possible trait values for each gene."""
    traits = {
        "conv1_neurons": [8, 16, 32, 64, 128, 256],
        "conv1_activation": ["relu", "selu", "sigmoid", "tanh"],
    }

    # Conv layers 2-5 (optional)
    for i in range(2, 6):
        traits[f"conv{i}_enabled"] = [True, False]
        traits[f"conv{i}_neurons"] = [8, 16, 32, 64, 128, 256]
        traits[f"conv{i}_activation"] = ["relu", "selu", "sigmoid", "tanh"]

    traits["convdense_dropout"] = [0, 0.2, 0.4, 0.5]

    # Dense layers 1-3 (optional)
    for i in range(1, 4):
        traits[f"dense{i}_enabled"] = [True, False]
        traits[f"dense{i}_neurons"] = [16, 32, 64, 128, 256, 512]
        traits[f"dense{i}_activation"] = ["relu", "selu", "sigmoid", "tanh"]
        traits[f"dense{i}_dropout"] = [0, 0.2, 0.4, 0.5]

    traits["final_activation"] = ["sigmoid", "softmax", "tanh"]
    traits["optimizer"] = ["adam", "sgd", "adagrad"]
    traits["loss"] = ["binary_crossentropy", "mse", "mae", "kld", "msle"]

    return traits


class GeneticConfig:
    """Configuration for genetic algorithm hyperparameter optimization."""

    # Possible values for each hyperparameter (the "gene pool")
    TRAITS_DICT = _build_traits_dict()

    # Population settings
    POPULATION_SIZE = 8
    FITTEST_RATIO = 0.25  # Top 25% become parents
    OGRES = 1  # Random "losers" kept for genetic diversity

    # Mutation settings
    DEFAULT_MUTATION_RATE = 15  # Percentage of traits that mutate

    # Target user IDs for fitness evaluation
    #TARGETS_TEST = [2, 3, 4, 5, 7, 8, 13, 15, 16, 19, 26, 28]
    TARGETS_TEST = [1, 2, 3, 4, 5]

    # Training settings for genome evaluation
    TRAIN_EPOCHS = 5
    TRAIN_SEGMENTS = [(1, 1), (1, 2), (1, 3), (1, 4)]  # (session_id, sequence_id)

    # Evolution settings
    GENERATIONS = 2  # Number of generations to evolve
    WINDOW_SIZE = 5  # Window size in seconds for data segmentation
    SIGNAL_TYPE = "bcg"  # Signal type: "bcg" or "bvp"

# =============================================================================
# Main Parameters Class
# =============================================================================

class Parameters:
    """
    Configuration container for the authentication system.

    Attributes:
        window_sz (int): Window size in seconds for data segmentation.
        hz (int): Sampling frequency in Hz.
        signal_type (SignalType): Type of signal processing (BVP or BCG).
        model (dict): CNN model architecture configuration.
        epochs (int): Number of training epochs.
        db (DB): Database connection for loading processed data.
    """

    def __init__(self, window_size=5, signal_type=SignalType.BCG, epochs=10):
        """
        Initialize parameters with specified configuration.

        Args:
            window_size (int): Window size in seconds. Defaults to 5.
            signal_type (SignalType): Signal processing type. Defaults to BCG.
        """
        # Data options
        self.window_sz = window_size
        self.hz = 50
        self.signal_type = signal_type

        # Model configurations
        self.epochs = epochs
        self.model = {
            'conv5_neurons': 128,
            'dense3_enabled': False,
            'dense2_dropout': 0.2,
            'dense1_dropout': 0.2,
            'conv5_enabled': False,
            'dense2_activation': 'sigmoid',
            'conv2_neurons': 256,
            'loss': 'mse',
            'convdense_dropout': 0.2,
            'conv3_enabled': True,
            'conv4_activation': 'selu',
            'optimizer': 'sgd',
            'final_activation': 'tanh',
            'conv3_neurons': 256,
            'conv2_activation': 'tanh',
            'dense1_enabled': True,
            'dense1_activation': 'tanh',
            'dense1_neurons': 512,
            'conv4_enabled': True,
            'conv5_activation': 'sigmoid',
            'conv1_neurons': 8,
            'conv4_neurons': 256,
            'dense2_enabled': False,
            'conv1_activation': 'tanh',
            'dense3_activation': 'tanh',
            'conv3_activation': 'selu',
            'dense3_neurons': 128,
            'conv2_enabled': True,
            'dense3_dropout': 0.5,
            'dense2_neurons': 128
        }

        # Get project root (parent of src/)
        src_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.dirname(src_dir)

        # Use signal-type-specific database directory
        db_dir_name = f"databases_{signal_type.value.upper()}"
        db_path = os.path.join(project_root, db_dir_name, f"db_{window_size}.sqlite")
        self.db = DB(db_path, init=False)
