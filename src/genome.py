"""
Genome representation for genetic algorithm-based neural architecture search.

Each genome encodes hyperparameters for a CNN model. Genomes can breed
to produce offspring with traits inherited from both parents, with mutation.
"""

from random import randint
from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Permute, Input,
    Activation, Dropout, Flatten, Dense, BatchNormalization
)

from .params import GeneticConfig


def random_select(options):
    """Select a random element from a list."""
    return options[randint(0, len(options) - 1)]


class Genome:
    """
    Represents a CNN architecture as a set of evolvable traits.

    Attributes:
        mutation_rate: Percentage of traits that mutate during breeding
        traits: Dict mapping trait names to their current values
    """

    def __init__(self, mutation_rate=None, traits=None, create=True):
        """
        Initialize a genome.

        Args:
            mutation_rate: Mutation rate for breeding
            traits: Pre-defined traits dict (for loading saved genomes)
            create: If True, generate random traits; if False, use provided traits
        """
        if create:
            self.mutation_rate = GeneticConfig.DEFAULT_MUTATION_RATE
            self.traits = {t: random_select(GeneticConfig.TRAITS_DICT[t]) for t in GeneticConfig.TRAITS_DICT}
        else:
            self.mutation_rate = int(mutation_rate)
            self.traits = traits

    def breed(self, other_genome):
        """
        Create a child genome by combining traits from two parents.

        For each trait, there's a (100 - mutation_rate)% chance to inherit
        from one of the parents, otherwise the child gets a random mutation.

        Args:
            other_genome: The other parent genome

        Returns:
            A new Genome with combined/mutated traits
        """
        child = Genome(mutation_rate=None, traits=None, create=True)
        for t in GeneticConfig.TRAITS_DICT:
            if randint(0, 100) > self.mutation_rate:
                # Inherit from one of the parents
                child.traits[t] = random_select([self.traits[t], other_genome.traits[t]])
            # else: keep the random mutation from __init__
        return child

    def generate_model(self, params):
        """
        Build a Keras CNN model from this genome's traits.

        Args:
            params: Parameters object with window_sz and hz

        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential()

        # Input and permute for proper dimension ordering
        model.add(Input(shape=(2, 3, params.window_sz * params.hz)))
        model.add(Permute((1, 3, 2)))  # (2, 3, N) -> (2, N, 3)

        # Conv block 1 (always present)
        model.add(Conv2D(self.traits["conv1_neurons"], (2, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation(self.traits["conv1_activation"]))
        model.add(MaxPooling2D(pool_size=(1, 2)))

        # Conv blocks 2-5 (optional)
        for i in range(2, 6):
            if self.traits[f"conv{i}_enabled"]:
                model.add(Conv2D(self.traits[f"conv{i}_neurons"], (2, 3), padding='same'))
                model.add(BatchNormalization())
                model.add(Activation(self.traits[f"conv{i}_activation"]))
                model.add(MaxPooling2D(pool_size=(1, 2)))

        if self.traits["convdense_dropout"] != 0:
            model.add(Dropout(self.traits["convdense_dropout"]))

        model.add(Flatten())

        # Dense blocks 1-3 (optional)
        for i in range(1, 4):
            if self.traits[f"dense{i}_enabled"]:
                model.add(Dense(self.traits[f"dense{i}_neurons"]))
                model.add(BatchNormalization())
                model.add(Activation(self.traits[f"dense{i}_activation"]))
                model.add(Dropout(self.traits[f"dense{i}_dropout"]))

        # Output layer
        model.add(Dense(1, kernel_initializer='normal'))
        model.add(BatchNormalization())
        model.add(Activation(self.traits["final_activation"]))

        model.compile(
            loss=self.traits["loss"],
            optimizer=self.traits["optimizer"],
            metrics=['accuracy']
        )
        return model
