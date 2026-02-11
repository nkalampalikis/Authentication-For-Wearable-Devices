"""
Test data preparation utilities for model evaluation.
"""

from .params import GeneticConfig


def prepare_first_session(target):
    """
    Prepare a test data set using the first session data.

    Args:
        target: The target user ID to authenticate

    Returns:
        A dict with 'positive' and 'negative' entries for testing.
        Each entry is a list of (user_id, (session_id, sequence_id)) tuples.
    """
    # Use session 1, sequence 5 for validation (not used in training)
    # Training uses session 1, sequences 1-4
    all_users = GeneticConfig.TARGETS_TEST

    positive = [(target, (1, 5))]
    negative = [(u, (1, 5)) for u in all_users if u != target]

    return {
        "positive": positive,
        "negative": negative
    }
