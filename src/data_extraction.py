"""
Data extraction utilities for loading and formatting sensor data from the database.

This module provides functions to:
- Load raw sensor data from SQLite databases
- Convert sensor data into numpy tensors for CNN input
- Create overlapping sequences for testing
"""

import json
import numpy as np


def load_from_db(db, subject_id, window_sz, session_id, sequence_id):
    """
    Load segment data from the database.

    Args:
        db: Database connection object
        subject_id: Integer ID of the subject/patient
        window_sz: Number of seconds of data in each segment
        session_id: Training session ID
        sequence_id: Sequence ID within the session

    Returns:
        List of segments, each containing:
        [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]
    """
    tables = db.cursor.execute("""
        SELECT table_name FROM data
        WHERE subject_id = ? AND window_sz = ? AND session_id = ?
        AND sequence_id = ?;
    """, [subject_id, window_sz, session_id, sequence_id])

    table = tables.fetchone()

    records = db.cursor.execute(f'''
        SELECT
            segment_gyro_x, segment_gyro_y, segment_gyro_z,
            segment_accl_x, segment_accl_y, segment_accl_z
        FROM `{table[0]}`;
    ''')

    return list(records.fetchall())


def extract_tensor(segment, window_sz, hz):
    """
    Convert a single segment into a CNN-ready tensor.

    Args:
        segment: List of 6 channels [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]
                 Each channel has window_sz * hz data points
        window_sz: Number of seconds of data
        hz: Sampling frequency (50 Hz)

    Returns:
        Numpy array of shape (1, 2, 3, window_sz*hz) where:
        - Dimension 0: batch (always 1)
        - Dimension 1: signal type (0=gyro, 1=accel)
        - Dimension 2: axis (x, y, z)
        - Dimension 3: time samples
    """
    samples = window_sz * hz

    # Stack gyro channels (indices 0-2) into shape (3, samples)
    gyro = np.stack([np.array(segment[i]) for i in range(3)], axis=0)

    # Stack accel channels (indices 3-5) into shape (3, samples)
    accel = np.stack([np.array(segment[i]) for i in range(3, 6)], axis=0)

    # Combine into (2, 3, samples) then add batch dimension
    tensor = np.stack([gyro, accel], axis=0)
    return tensor.reshape(1, 2, 3, samples)


def sequence_to_nparray(sequence, window_sz, hz):
    """
    Convert a sequence of database records into numpy arrays.

    Args:
        sequence: List of segments from load_from_db()
        window_sz: Number of seconds of data per segment
        hz: Sampling frequency

    Returns:
        List of numpy arrays, each with shape (1, 2, 3, window_sz*hz)
    """
    tensors = []

    for segment in sequence:
        # Parse JSON-encoded channel data
        channels = [json.loads(channel)[:window_sz * hz] for channel in segment]
        tensor = extract_tensor(channels, window_sz, hz)
        tensors.append(tensor)

    return tensors


def collect_segment_data(params, person, train_segs):
    """
    Collect and format segment data for a person.

    Args:
        params: Parameters object containing window_sz, hz, and db
        person: Person/subject ID
        train_segs: List of (session_id, sequence_id) tuples

    Returns:
        List of numpy arrays ready for model input,
        each with shape (1, 2, 3, window_sz*hz)
    """
    points = []

    for session_id, sequence_id in train_segs:
        sequence = load_from_db(
            params.db, person, params.window_sz, session_id, sequence_id
        )
        tensors = sequence_to_nparray(sequence, params.window_sz, params.hz)
        points.extend(tensors)

    return points


def make_sequential(params, data, sequence_length):
    """
    Create overlapping sequences from data for improved prediction accuracy.

    Takes a list of segments and creates sliding windows of length sequence_length,
    with overlap of (sequence_length - 1). This increases the effective training
    data and improves CNN accuracy.

    Example with sequence_length=3:
        Input:  [seg1, seg2, seg3, seg4, seg5]
        Output: [[seg1, seg2, seg3],
                 [seg2, seg3, seg4],
                 [seg3, seg4, seg5]]

    Args:
        params: Parameters object
        data: List of data segments
        sequence_length: Number of segments per sequence

    Returns:
        List of numpy arrays, each with shape
        (sequence_length, 2, 3, window_sz*hz)
    """
    if len(data) < sequence_length:
        return []

    sequences = []
    samples = params.window_sz * params.hz

    for i in range(len(data) - sequence_length + 1):
        seq_tensor = np.zeros((sequence_length, 2, 3, samples))

        for k in range(sequence_length):
            seq_tensor[k] = data[i + k]

        sequences.append(seq_tensor)

    return sequences
