#!/usr/bin/env python3
"""
Command Line Interface for Physiological Authentication System

Usage:
    python main.py train --targets 9 10 11 --window-size 4 --epochs 100
    python main.py train --targets 9 10 --signal-type bvp
    python main.py test --targets 9 10 --signal-type bvp
    python main.py genetic
    python main.py init-db
    python main.py init-db --signal-type bvp --window-sizes 4 5
    python main.py plot-training --log-dir ./log_training --targets 2 3 4 5
    python main.py parse-logs --genetic-dir ./genetic_bcg_5
"""

import argparse
from copy import deepcopy
from pathlib import Path

from src import Parameters, Trainer, Model, test, SignalType
from src.genetic import main as genetic_main
from src.loader import load_db
from src.test_data import prepare_first_session
from src.visualization import plot_training_curves, parse_genetic_logs


def get_signal_type(signal_type_str):
    """Convert string to SignalType enum."""
    if signal_type_str.lower() == "bvp":
        return SignalType.BVP
    return SignalType.BCG


def train_command(args):
    """Train models for specified users."""
    signal_type = get_signal_type(args.signal_type)
    p = Parameters(args.window_size, signal_type)
    p.epochs = args.epochs

    targets = args.targets
    print(f"Training models for users: {targets}")
    print(f"Window size: {args.window_size}s, Epochs: {args.epochs}")
    print(f"Signal type: {signal_type.value.upper()}")

    for owner in targets:
        print(f"\n{'='*50}")
        print(f"Training model for user {owner}")
        print(f"{'='*50}")
        others = deepcopy(targets)
        others.remove(owner)
        t = Trainer(p, owner, others)
        t.train(p)

    print("\nTraining complete!")


def test_command(args):
    """Test trained models for specified users."""
    signal_type = get_signal_type(args.signal_type)
    p = Parameters(args.window_size, signal_type)

    targets = args.targets
    print(f"Testing models for users: {targets}")
    print(f"Window size: {args.window_size}s, Threshold: {args.threshold}")
    print(f"Signal type: {signal_type.value.upper()}")

    results_summary = []

    for owner in targets:
        print(f"\n{'='*50}")
        print(f"Testing model for user {owner}")
        print(f"{'='*50}")

        # Load the trained model
        model_dir = Path("models_cache") / str(args.window_size) / str(owner)
        topo_path = str(model_dir / "topo.json")
        weights_path = str(model_dir / "model.weights.h5")

        try:
            model = Model.load_model(p, topo_path, weights_path)
        except FileNotFoundError:
            print(f"Model not found for user {owner}. Run train first.")
            continue

        # Prepare test data
        data_set = prepare_first_session(owner)

        # Run test
        results = test(p, model, data_set, args.threshold, sequence_length=1)

        tar = results["tar"]
        trr = results["trr"]
        far = 100 - trr
        frr = 100 - tar

        print(f"\nResults for user {owner}:")
        print(f"  TAR (True Accept Rate):  {tar:.2f}%")
        print(f"  TRR (True Reject Rate):  {trr:.2f}%")
        print(f"  FAR (False Accept Rate): {far:.2f}%")
        print(f"  FRR (False Reject Rate): {frr:.2f}%")

        results_summary.append({
            "user": owner,
            "tar": tar,
            "trr": trr,
            "far": far,
            "frr": frr
        })

    # Print summary
    if results_summary:
        print(f"\n{'='*50}")
        print("Summary")
        print(f"{'='*50}")
        avg_tar = sum(r["tar"] for r in results_summary) / len(results_summary)
        avg_trr = sum(r["trr"] for r in results_summary) / len(results_summary)
        avg_far = sum(r["far"] for r in results_summary) / len(results_summary)
        avg_frr = sum(r["frr"] for r in results_summary) / len(results_summary)
        print(f"Average TAR: {avg_tar:.2f}%")
        print(f"Average TRR: {avg_trr:.2f}%")
        print(f"Average FAR: {avg_far:.2f}%")
        print(f"Average FRR: {avg_frr:.2f}%")

    print("\nTesting complete!")


def genetic_command(_args):
    """Run genetic algorithm optimization."""
    genetic_main()


def init_db_command(args):
    """Initialize databases from raw sensor data."""
    signal_types = [get_signal_type(args.signal_type)] if args.signal_type else [SignalType.BCG, SignalType.BVP]
    window_sizes = args.window_sizes if args.window_sizes else [2, 3, 4, 5]

    for signal_type in signal_types:
        for ws in window_sizes:
            print(f"\n=== Initializing {signal_type.value.upper()} database for window size {ws} ===")
            load_db(ws, signal_type)

    print("\nDatabase initialization complete!")


def plot_training_command(args):
    """Plot training accuracy curves."""
    plot_training_curves(
        log_dir=args.log_dir,
        targets=args.targets,
        output_path=args.output
    )


def parse_logs_command(args):
    """Parse genetic algorithm logs and show statistics."""
    parse_genetic_logs(
        genetic_dir=args.genetic_dir,
        output_path=args.output
    )


def main():
    parser = argparse.ArgumentParser(
        description="Physiological Signal-Based User Authentication System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --targets 9 10 11 12 --window-size 4
  python main.py train --targets 9 10 --signal-type bvp
  python main.py test --targets 9 10 --signal-type bvp --threshold 0.5
  python main.py genetic
  python main.py init-db
  python main.py init-db --signal-type bvp --window-sizes 4 5
  python main.py plot-training --log-dir ./log_training --targets 2 3 4
  python main.py parse-logs --genetic-dir ./genetic_bcg_5
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train authentication models for specified users"
    )
    train_parser.add_argument(
        "--targets", "-t",
        nargs="+",
        type=int,
        default=[9, 10, 11, 12, 14, 24],
        help="User IDs to train models for (default: 9 10 11 12 14 24)"
    )
    train_parser.add_argument(
        "--window-size", "-w",
        type=int,
        default=4,
        help="Window size in seconds (default: 4)"
    )
    train_parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    train_parser.add_argument(
        "--signal-type", "-s",
        choices=["bvp", "bcg"],
        default="bcg",
        help="Signal type to use: bvp or bcg (default: bcg)"
    )

    # Test command
    test_parser = subparsers.add_parser(
        "test",
        help="Test trained authentication models for specified users"
    )
    test_parser.add_argument(
        "--targets", "-t",
        nargs="+",
        type=int,
        default=[9, 10, 11, 12, 14, 24],
        help="User IDs to test models for (default: 9 10 11 12 14 24)"
    )
    test_parser.add_argument(
        "--window-size", "-w",
        type=int,
        default=4,
        help="Window size in seconds (default: 4)"
    )
    test_parser.add_argument(
        "--threshold", "-th",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)"
    )
    test_parser.add_argument(
        "--signal-type", "-s",
        choices=["bvp", "bcg"],
        default="bcg",
        help="Signal type to use: bvp or bcg (default: bcg)"
    )

    # Genetic command (all settings from GeneticConfig in params.py)
    subparsers.add_parser(
        "genetic",
        help="Run genetic algorithm for hyperparameter optimization"
    )

    # Init-db command
    init_db_parser = subparsers.add_parser(
        "init-db",
        help="Initialize SQLite databases from raw sensor data"
    )
    init_db_parser.add_argument(
        "--window-sizes", "-w",
        nargs="+",
        type=int,
        help="Window sizes to initialize (default: 2 3 4 5)"
    )
    init_db_parser.add_argument(
        "--signal-type", "-s",
        choices=["bvp", "bcg"],
        help="Signal type to process: bvp or bcg (default: both)"
    )

    # Plot training command
    plot_training_parser = subparsers.add_parser(
        "plot-training",
        help="Plot training accuracy curves from log files"
    )
    plot_training_parser.add_argument(
        "--log-dir", "-d",
        required=True,
        help="Directory containing training log files"
    )
    plot_training_parser.add_argument(
        "--targets", "-t",
        nargs="+",
        type=int,
        required=True,
        help="User IDs to plot"
    )
    plot_training_parser.add_argument(
        "--output", "-o",
        help="Output file path (if not specified, displays interactively)"
    )

    # Parse logs command
    parse_logs_parser = subparsers.add_parser(
        "parse-logs",
        help="Parse genetic algorithm logs and show statistics"
    )
    parse_logs_parser.add_argument(
        "--genetic-dir", "-d",
        required=True,
        help="Directory containing genetic algorithm results (e.g., ./genetic_5)"
    )
    parse_logs_parser.add_argument(
        "--output", "-o",
        help="Output file path for box plot (if not specified, displays interactively)"
    )

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "test":
        test_command(args)
    elif args.command == "genetic":
        genetic_command(args)
    elif args.command == "init-db":
        init_db_command(args)
    elif args.command == "plot-training":
        plot_training_command(args)
    elif args.command == "parse-logs":
        parse_logs_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
