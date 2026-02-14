"""
Visualization utilities for model evaluation and analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def plot_roc_curves(params, model_loader, targets, test_func, output_dir=None):
    """
    Generate ROC curves for all target users.

    Args:
        params: Parameters object
        model_loader: Function that takes (params, target) and returns a model
        targets: List of target user IDs
        test_func: Function that takes (params, model, target) and returns [tar, trr, y_pred, y_true]
        output_dir: Directory to save plots (if None, displays interactively)
    """
    line_styles = ["-", "--", "-.", ":"]

    plt.figure(figsize=(10, 8))
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.xlabel('False Accept Rate (%)')
    plt.ylabel('True Accept Rate (%)')

    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    legends = []

    for idx, target in enumerate(targets):
        print(f"Processing target {target}...")
        model = model_loader(params, target)
        results = test_func(params, model, target)

        y_true = results[3]
        y_pred = results[2]

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        test_auc = auc(fpr, tpr)

        plt.plot(
            100 * fpr, 100 * tpr,
            linestyle=line_styles[idx % len(line_styles)],
            label=f'User {target}: AUC {test_auc:.4f}, EER {eer*100:.2f}%'
        )
        legends.append((test_auc, eer * 100))

    # Plot average
    tprs[-1][0] = 0.0
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    eer = brentq(lambda x: 1. - x - interp1d(mean_fpr, mean_tpr)(x), 0., 1.)
    test_auc = auc(mean_fpr, mean_tpr)

    plt.plot(
        mean_fpr * 100, mean_tpr * 100,
        lw=3, color='black', linestyle='dotted',
        label=f'Average: AUC {test_auc:.4f}, EER {eer*100:.2f}%'
    )

    plt.legend(loc='lower right')
    signal_name = params.signal_type.value.upper()
    plt.title(f'ROC Curves ({signal_name}, Window: {params.window_sz}s)')
    plt.grid(True, alpha=0.3)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150)
        print(f"Saved to {output_dir}/roc_curves.png")
    else:
        plt.show()


def plot_training_curves(log_dir, targets, output_path=None):
    """
    Plot training accuracy curves from log files.

    Args:
        log_dir: Directory containing training log files ({target}.log)
        targets: List of target user IDs
        output_path: Path to save plot (if None, displays interactively)
    """
    line_styles = ["-", "--", "-.", ":"]

    plt.figure(figsize=(10, 6))
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy (%)')

    for idx, target in enumerate(targets):
        log_path = os.path.join(log_dir, f"{target}.log")

        if not os.path.exists(log_path):
            print(f"Warning: Log file not found for target {target}")
            continue

        with open(log_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if '=' in line:
                nums = line.split('=')[1].strip()
                nums = nums.replace('[', '').replace(']', '')
                accuracy = [float(x) * 100 for x in nums.split(',')]
                epochs = list(range(1, len(accuracy) + 1))

                plt.plot(
                    epochs, accuracy,
                    linestyle=line_styles[idx % len(line_styles)],
                    label=f'User {target}'
                )

    plt.legend()
    plt.title('Training Accuracy Curves')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, None])
    plt.ylim([0, 100])

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved to {output_path}")
    else:
        plt.show()


def parse_genetic_logs(genetic_dir, output_path=None):
    """
    Parse genetic algorithm logs and create box plots of scores by generation.

    Args:
        genetic_dir: Directory containing generation subdirectories (gen0, gen1, ...)
        output_path: Path to save plot (if None, displays interactively)
    """
    scores_by_gen = {}

    for root, files in os.walk(genetic_dir):
        for file in files:
            if file.endswith('.log') and 'parent' not in file and 'ogre' not in file:
                filepath = os.path.join(root, file)
                gen_name = os.path.basename(root)

                try:
                    with open(filepath, 'r') as f:
                        lines = f.readlines()

                    if lines:
                        last_line = lines[-1].strip()
                        if 'Score:' in last_line:
                            score = float(last_line.split(':')[1].strip())
                        else:
                            score = float(last_line)

                        if gen_name not in scores_by_gen:
                            scores_by_gen[gen_name] = []
                        scores_by_gen[gen_name].append(score)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse {filepath}: {e}")

    if not scores_by_gen:
        print("No valid log files found")
        return

    # Sort generations
    sorted_gens = sorted(scores_by_gen.keys(), key=lambda x: int(x.replace('gen', '')))
    data = [scores_by_gen[gen] for gen in sorted_gens]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=sorted_gens)
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('Genetic Algorithm Score Distribution by Generation')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved to {output_path}")
    else:
        plt.show()

    # Print summary statistics
    print("\nGeneration Statistics:")
    print("-" * 50)
    for gen in sorted_gens:
        scores = scores_by_gen[gen]
        print(f"{gen}: min={min(scores):.2f}, max={max(scores):.2f}, "
              f"mean={np.mean(scores):.2f}, median={np.median(scores):.2f}")
