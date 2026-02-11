"""
Model management for the authentication CNN.

This module handles building, saving, and loading Keras models
with configurable architecture defined by params.model dict.
"""

from pathlib import Path

from keras.models import Sequential, model_from_json
from keras.layers import (
    Conv2D, MaxPooling2D, Permute, Input,
    Activation, Dropout, Flatten, Dense, BatchNormalization
)


class Model:
    """Static utility class for CNN model operations."""

    @staticmethod
    def get(params, topo_path, weights_path):
        """
        Get a model - load from disk if exists, otherwise build new.

        Args:
            params: Parameters object with model config
            topo_path: Path to model architecture JSON
            weights_path: Path to model weights H5

        Returns:
            Compiled Keras model
        """
        if Path(topo_path).is_file() and Path(weights_path).is_file():
            print("model loaded from disk")
            return Model.load_model(params, topo_path, weights_path)
        return Model.build_model(params)

    @staticmethod
    def save_model(model, topo_path, weights_path):
        """
        Save model architecture and weights to disk.

        Args:
            model: Keras model to save
            topo_path: Path to save architecture JSON
            weights_path: Path to save weights H5
        """
        # Create directories if needed
        Path(topo_path).parent.mkdir(parents=True, exist_ok=True)
        Path(weights_path).parent.mkdir(parents=True, exist_ok=True)

        # Save architecture as JSON
        with open(topo_path, 'w') as f:
            f.write(model.to_json())

        # Save weights
        model.save_weights(weights_path)

    @staticmethod
    def load_model(params, topo_path, weights_path):
        """
        Load a previously saved model from disk.

        Args:
            params: Parameters object with model config for compilation
            topo_path: Path to model architecture JSON
            weights_path: Path to model weights H5

        Returns:
            Compiled Keras model
        """
        with open(topo_path, 'r') as f:
            model_json = f.read()

        model = model_from_json(model_json)
        model.load_weights(weights_path)

        # Compile with same settings as build_model
        model.compile(
            loss=params.model["loss"],
            optimizer=params.model["optimizer"],
            metrics=['accuracy']
        )
        return model

    @staticmethod
    def build_model(params):
        """
        Build a new CNN model from params configuration.

        Args:
            params: Parameters object containing:
                - window_sz: Window size in seconds
                - hz: Sampling frequency
                - model: Dict with layer configs

        Returns:
            Compiled Keras model

        Model architecture:
            Input: (2, 3, window_sz * hz) - 2 signal types, 3 axes, N samples
            → Permute to (2, N, 3)
            → 1-5 Conv2D blocks (configurable)
            → Optional dropout
            → Flatten
            → 0-3 Dense blocks (configurable)
            → Dense(1) output with final activation
        """
        model = Sequential()
        config = params.model

        # Input and permute for proper dimension ordering
        model.add(Input(shape=(2, 3, params.window_sz * params.hz)))
        model.add(Permute((1, 3, 2)))  # (2, 3, N) -> (2, N, 3)

        # Conv block 1 (always present)
        model.add(Conv2D(int(config["conv1_neurons"]), (2, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation(config["conv1_activation"]))
        model.add(MaxPooling2D(pool_size=(1, 2)))

        # Conv blocks 2-5 (optional)
        for i in range(2, 6):
            if config[f"conv{i}_enabled"]:
                model.add(Conv2D(config[f"conv{i}_neurons"], (2, 3), padding='same'))
                model.add(BatchNormalization())
                model.add(Activation(config[f"conv{i}_activation"]))
                model.add(MaxPooling2D(pool_size=(1, 2)))

        # Dropout between conv and dense
        if config["convdense_dropout"] != 0:
            model.add(Dropout(config["convdense_dropout"]))

        model.add(Flatten())

        # Dense blocks 1-3 (optional)
        for i in range(1, 4):
            if config[f"dense{i}_enabled"]:
                model.add(Dense(config[f"dense{i}_neurons"]))
                model.add(BatchNormalization())
                model.add(Activation(config[f"dense{i}_activation"]))
                model.add(Dropout(config[f"dense{i}_dropout"]))

        # Output layer
        model.add(Dense(1, kernel_initializer='normal'))
        model.add(BatchNormalization())
        model.add(Activation(config["final_activation"]))

        model.compile(
            loss=config["loss"],
            optimizer=config["optimizer"],
            metrics=['accuracy']
        )
        return model
