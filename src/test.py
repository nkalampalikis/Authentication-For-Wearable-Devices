"""
Testing module for evaluating authentication model performance.

This module provides functions to test trained models and compute
authentication metrics like TAR (True Accept Rate) and TRR (True Reject Rate).
"""

from sklearn.metrics import confusion_matrix

from .data_extraction import collect_segment_data, make_sequential


def test(params, model, data_set, threshold, sequence_length, verbose=True):
    """
    Evaluate model performance on a test dataset.

    Args:
        params: Parameters object with window_sz, hz, and db
        model: Trained Keras model to evaluate
        data_set: Dict with 'positive' and 'negative' entries, each containing
                  list of (user_id, (session_id, sequence_id)) tuples
        threshold: Classification threshold (predictions > threshold → accept)
        sequence_length: Number of segments per sequence for prediction
        verbose: If True, print progress messages (default: True)

    Returns:
        Dict with keys:
            - tar: True Accept Rate (% of positives correctly accepted)
            - trr: True Reject Rate (% of negatives correctly rejected)
            - predictions: List of raw prediction values
            - labels: List of true labels
    """
    if verbose:
        _print_test_summary(params, data_set, threshold, sequence_length)
        print("Loading testing data...", end='', flush=True)

    sequences, labels = _load_test_data(params, data_set, sequence_length)

    if verbose:
        print("done")
        print("Testing...", end='', flush=True)

    predictions, results = _predict(model, sequences, threshold)

    if verbose:
        print("done\n")

    # Calculate metrics
    assert len(results) == len(labels)
    tn, fp, fn, tp = confusion_matrix(labels, results, labels=[0, 1]).ravel()

    tar = (tp / (tp + fn)) * 100.0  # True Accept Rate
    trr = (tn / (tn + fp)) * 100.0  # True Reject Rate

    return {
        "tar": tar,
        "trr": trr,
        "predictions": predictions,
        "labels": labels
    }


def _print_test_summary(params, data_set, threshold, sequence_length):
    """Print test configuration summary."""
    print("---------------------------------")
    print("Test Parameter Summary:")
    print("---------------------------------")
    print(f"Threshold:\t{threshold:.2f}\tSeq. Len.:\t{sequence_length}")
    print(f"Win. Size:\t{params.window_sz}\tData Freq.:\t{params.hz}")
    print("Positive Data Set:")
    print(data_set["positive"])
    print("Negative Data Set:")
    print(data_set["negative"])
    print("=================================")


def _load_test_data(params, data_set, sequence_length):
    """
    Load and prepare test data from dataset specification.

    Args:
        params: Parameters object
        data_set: Dict with 'positive' and 'negative' entries
        sequence_length: Number of segments per sequence

    Returns:
        Tuple of (sequences, labels) lists
    """
    sequences = []
    labels = []

    # Load negative samples (imposters)
    for user_id, segment in data_set["negative"]:
        points = collect_segment_data(params, user_id, [segment])
        seqs = make_sequential(params, points, sequence_length)
        sequences.extend(seqs)
        labels.extend([0] * len(seqs))

    # Load positive samples (authentic user)
    for user_id, segment in data_set["positive"]:
        points = collect_segment_data(params, user_id, [segment])
        seqs = make_sequential(params, points, sequence_length)
        sequences.extend(seqs)
        labels.extend([1] * len(seqs))

    return sequences, labels


def _predict(model, sequences, threshold):
    """
    Run predictions on sequences.

    Args:
        model: Keras model
        sequences: List of sequence tensors
        threshold: Classification threshold

    Returns:
        Tuple of (raw_predictions, binary_results) lists
    """
    predictions = []
    results = []

    for seq in sequences:
        pred = model.predict(seq, verbose=0)
        predictions.append(float(pred[0][0]))

        # Apply threshold: if any prediction in sequence > threshold, accept
        binary = (pred > threshold).astype(int)
        results.append(1 if 1 in binary else 0)

    return predictions, results
