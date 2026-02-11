"""
Genetic algorithm for CNN hyperparameter optimization.

This module evolves a population of neural network architectures to find
optimal hyperparameters for user authentication.
"""

import ast
import csv
from copy import deepcopy
from pathlib import Path
from random import shuffle, randint

import numpy as np

from .params import Parameters, GeneticConfig
from .genome import Genome
from .data_extraction import collect_segment_data
from .test import test
from .test_data import prepare_first_session


def run_model(params, model, threshold, sequence_length, target):
    """
    Train and test a model for a specific target user.

    Args:
        params: Parameters object
        model: Keras model to train
        threshold: Classification threshold
        sequence_length: Sequence length for testing
        target: Target user ID

    Returns:
        Dict with tar, trr, predictions, labels
    """
    data_set = prepare_first_session(target)

    others = [u for u in GeneticConfig.TARGETS_TEST if u != target]

    # Collect training data
    pos = collect_segment_data(params, target, GeneticConfig.TRAIN_SEGMENTS)
    neg = []
    for other in others:
        neg.extend(collect_segment_data(params, other, GeneticConfig.TRAIN_SEGMENTS))

    # Build training tensor
    n_samples = len(pos) + len(neg)
    train_tensor = np.zeros((n_samples, 2, 3, params.window_sz * params.hz))

    for i, sample in enumerate(neg):
        train_tensor[i] = sample
    for i, sample in enumerate(pos):
        train_tensor[len(neg) + i] = sample

    train_labels = np.array([0] * len(neg) + [1] * len(pos))

    # Class weights for imbalanced data
    weights = {0: 1.0, 1: len(neg) / len(pos)}

    # Train
    model.fit(
        train_tensor, train_labels,
        batch_size=32,
        epochs=GeneticConfig.TRAIN_EPOCHS,
        verbose=0,
        class_weight=weights,
        shuffle=True
    )

    # Test
    return test(params, model, data_set, threshold, sequence_length, verbose=False)


def evaluate_genome(genome, params, log_file, roc_file):
    """
    Evaluate a genome's fitness across all test users.

    Args:
        genome: Genome to evaluate
        params: Parameters object
        log_file: Path to write log results
        roc_file: Path to write prediction data

    Returns:
        float: Fitness score (lower is better)
    """
    all_predictions = []
    all_labels = []
    targets = GeneticConfig.TARGETS_TEST

    with open(log_file, 'w') as f:
        f.write(f"{genome.traits}\n")
        f.write("ID\tTAR\tTRR\n")

        total_tar, total_trr = 0, 0
        total_far, total_frr = 0, 0

        for target_id in targets:
            model = genome.generate_model(params)
            print(f"  Target {target_id}...", end="", flush=True)

            results = run_model(params, model, 0.5, 1, target_id)
            tar, trr = results["tar"], results["trr"]
            far, frr = 100 - trr, 100 - tar

            print(f" TAR={tar:.1f}%, TRR={trr:.1f}%")

            total_tar += tar
            total_trr += trr
            total_far += far
            total_frr += frr

            f.write(f"{target_id}\t{tar:.2f}\t{trr:.2f}\n")
            all_predictions.extend(results["predictions"])
            all_labels.extend(results["labels"])

        # Compute averages
        n = len(targets)
        avg_tar, avg_trr = total_tar / n, total_trr / n
        avg_far, avg_frr = total_far / n, total_frr / n

        # Fitness score: minimize FAR and FRR
        score = (avg_far ** 2) + (avg_frr ** 2)

        f.write(f"Avg TAR: {avg_tar:.2f}\n")
        f.write(f"Avg TRR: {avg_trr:.2f}\n")
        f.write(f"Score: {score:.2f}\n")

    # Write predictions for ROC analysis
    with open(roc_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prediction', 'label'])
        writer.writerows(zip(all_predictions, all_labels))

    return score


def get_generation_count(signal_type, window_size):
    """Get the number of completed generations."""
    gen_dir = Path(f"./genetic_{signal_type}_{window_size}")
    if not gen_dir.is_dir():
        return 0
    return len(list(gen_dir.iterdir()))


def save_generation(generation, generation_index, signal_type, window_size):
    """Save a generation to disk for later resumption."""
    save_dir = Path(f"./saved_gens_{signal_type}_{window_size}")
    save_dir.mkdir(parents=True, exist_ok=True)

    gen_data = [
        [score, genome.mutation_rate, genome.traits]
        for score, genome in generation
    ]

    save_file = save_dir / f"saved_generation{generation_index}.csv"
    with open(save_file, "w", newline='') as f:
        csv.writer(f).writerows(gen_data)

    print(f"Saved generation {generation_index} ({len(gen_data)} genomes)")


def load_generation(generation_index, signal_type, window_size):
    """Load a saved generation from disk."""
    save_file = Path(f"./saved_gens_{signal_type}_{window_size}/saved_generation{generation_index - 1}.csv")

    if not save_file.exists():
        raise FileNotFoundError(
            f"No saved generation at {save_file}. "
            f"Delete genetic_* directories to start fresh."
        )

    generation = []
    with open(save_file, "r") as f:
        for row in csv.reader(f):
            score = float(row[0])
            traits = ast.literal_eval(row[2])
            genome = Genome(mutation_rate=row[1], traits=traits, create=False)
            generation.append([score, genome])

    return generation


def create_seed_generation(params, gen_dir):
    """Create and evaluate the initial random population."""
    population_size = GeneticConfig.POPULATION_SIZE
    generation = []

    gen_dir.mkdir(parents=True, exist_ok=True)

    for i in range(population_size):
        print(f"\nGenome {i + 1}/{population_size}")
        genome = Genome(mutation_rate=None, traits=None, create=True)

        log_file = gen_dir / f"{i}.log"
        roc_file = gen_dir / f"{i}.csv"
        score = evaluate_genome(genome, params, log_file, roc_file)

        generation.append([score, genome])
        print(f"Score: {score:.2f}")

    return generation


def select_parents(generation, population_size, fittest_ratio, ogres):
    """Select parents for the next generation."""
    # Sort by fitness (lower score = better)
    sorted_gen = sorted(generation, key=lambda x: x[0])

    # Take the fittest
    n_fittest = round(population_size * fittest_ratio)
    parents = sorted_gen[:n_fittest]

    # Add random "ogres" for genetic diversity
    remaining = sorted_gen[n_fittest:]
    shuffle(remaining)
    parents.extend(remaining[:ogres])

    return parents


def breed_generation(parents, params, population_size, gen_dir):
    """Create a new generation by breeding parents."""
    # Start with parents (they survive to next generation)
    new_generation = deepcopy(parents)
    child_index = 0

    while len(new_generation) < population_size:
        # Select two different parents
        p1, p2 = randint(0, len(parents) - 1), randint(0, len(parents) - 1)
        if p1 == p2:
            continue

        # Breed and evaluate child
        child = parents[p1][1].breed(parents[p2][1])

        print(f"\nChild {child_index + 1}")
        log_file = gen_dir / f"{child_index}.log"
        roc_file = gen_dir / f"{child_index}.csv"
        score = evaluate_genome(child, params, log_file, roc_file)

        new_generation.append([score, child])
        print(f"Score: {score:.2f}")
        child_index += 1

    return new_generation


def main():
    """Run the genetic algorithm for hyperparameter optimization."""
    import warnings
    warnings.filterwarnings('ignore', message='.*softmax.*')

    from .signal_filter import SignalType

    window_size = GeneticConfig.WINDOW_SIZE
    generations = GeneticConfig.GENERATIONS
    signal_type_str = GeneticConfig.SIGNAL_TYPE
    signal_type = SignalType.BVP if signal_type_str == "bvp" else SignalType.BCG

    params = Parameters(window_size, signal_type)
    population_size = GeneticConfig.POPULATION_SIZE
    fittest_ratio = GeneticConfig.FITTEST_RATIO
    ogres = GeneticConfig.OGRES

    # Directory naming: genetic_{signal_type}_{window_size}
    base_dir = f"genetic_{signal_type_str}_{window_size}"

    print("Genetic Algorithm Optimization")
    print(f"Signal type: {signal_type_str.upper()}, Window size: {window_size}s, Generations: {generations}")
    print(f"Population: {population_size}, Fittest: {fittest_ratio*100:.0f}%, Ogres: {ogres}")

    # Check for existing progress
    start_gen = get_generation_count(signal_type_str, window_size)
    end_gen = start_gen + generations

    # Verify we can resume if there's existing progress
    if start_gen > 0:
        save_file = Path(f"./saved_gens_{signal_type_str}_{window_size}/saved_generation{start_gen - 1}.csv")
        if not save_file.exists():
            print(f"\nWarning: Found {base_dir}/ directory but no save file.")
            print("Starting fresh from generation 0.")
            start_gen = 0
            end_gen = generations

    # Create seed generation if starting fresh
    if start_gen == 0:
        print("\n=== Creating seed generation (gen0) ===")
        gen_dir = Path(f"./{base_dir}/gen0")
        current_generation = create_seed_generation(params, gen_dir)
        save_generation(current_generation, 0, signal_type_str, window_size)
        start_gen = 1

    # Evolve for remaining generations
    for gen_index in range(start_gen, end_gen):
        print(f"\n=== Generation {gen_index} ===")

        # Load previous generation
        current_generation = load_generation(gen_index, signal_type_str, window_size)
        gen_dir = Path(f"./{base_dir}/gen{gen_index}")
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Select parents
        parents = select_parents(
            current_generation, population_size, fittest_ratio, ogres
        )
        print(f"Selected {len(parents)} parents")

        # Log parent scores
        for i, (score, _) in enumerate(parents[:round(population_size * fittest_ratio)]):
            (gen_dir / f"parent_{i}.log").write_text(f"{score}\n")
        for i, (score, _) in enumerate(parents[round(population_size * fittest_ratio):]):
            (gen_dir / f"ogre_{i}.log").write_text(f"{score}\n")

        # Breed new generation
        current_generation = breed_generation(
            parents, params, population_size, gen_dir
        )
        save_generation(current_generation, gen_index, signal_type_str, window_size)

    # Report best genome
    best_score, best_genome = min(current_generation, key=lambda x: x[0])
    print(f"\n{'='*50}")
    print(f"Best genome (score: {best_score:.2f}):")
    print(best_genome.traits)


if __name__ == "__main__":
    main()
