"""
Signal processing filters for physiological data.

This module provides bandpass filtering for extracting BVP (Blood Volume Pulse)
and BCG (Ballistocardiography) signals from raw accelerometer and gyroscope data.

BVP signals are typically in the 0.75-2.5 Hz range and represent blood flow.
BCG signals are typically in the 4-11 Hz range and represent body motion from heartbeats.
"""

from scipy.stats import zscore
import numpy as np
from scipy.signal import butter, lfilter
from enum import Enum


class SignalType(Enum):
    """Enumeration of supported physiological signal types."""
    BVP = "bvp"
    BCG = "bcg"


# Filter parameters for each signal type
FILTER_PARAMS = {
    SignalType.BVP: {
        "rolling_avg_window": 3,
        "bandpass_low": 0.75,
        "bandpass_high": 2.5,
        "bandpass_order": 2,
    },
    SignalType.BCG: {
        "rolling_avg_window": 35,
        "bandpass_low": 4,
        "bandpass_high": 11,
        "bandpass_order": 4,
    },
}


class SignalFilter:
    """
    Signal filter for extracting physiological signals from raw sensor data.

    This class processes raw accelerometer and gyroscope data through a pipeline of:
    1. Z-score normalization
    2. Rolling average filter (removes baseline drift)
    3. Butterworth bandpass filter (isolates frequency range of interest)
    4. Min-max normalization

    Attributes:
        hz (int): Sampling frequency in Hz.
        signal_type (SignalType): Type of signal being extracted (BVP or BCG).
        filtered_data (dict): Dictionary of filtered signal components.

    """


    def __init__(self, raw, hz, signal_type):
        """
        Initialize the signal filter and process raw data.

        Args:
            raw: Structured numpy array with fields 'TS', 'gyro_X', 'gyro_Y',
                 'gyro_Z', 'accel_X', 'accel_Y', 'accel_Z'.
            hz (int): Sampling frequency in Hz (typically 50).
            signal_type (SignalType): Type of signal to extract.
        """
        self.hz = hz
        self.signal_type = signal_type
        self.filtered_data = {}

        # Get filter parameters for this signal type
        params = FILTER_PARAMS[signal_type]

        # Set up intermediate arrays
        data_zscore = {}
        data_avg = {}
        data_butter = {}
        data_norm = {}

        # For each component (gyro_{x,y,z}, accel_{x,y,z})
        for v in list(filter((lambda x: x != "TS"), raw.dtype.names)):
            # Step 1: Z-score normalization
            data_zscore[v] = zscore(raw[v])

            # Step 2: Rolling average filter (removes baseline drift)
            data_avg[v] = self._rolling_avg_filter(
                data_zscore[v],
                params["rolling_avg_window"]
            )

            # Step 3: Butterworth bandpass filter
            data_butter[v] = self._butter_bandpass_filter(
                data_avg[v],
                params["bandpass_low"],
                params["bandpass_high"],
                hz,
                params["bandpass_order"]
            )

            # Step 4: Min-max normalization
            data_norm[v] = self._normalize(data_butter[v])

            # Store final result
            self.filtered_data[v] = self._clean(data_norm[v])

    # Backward compatibility: expose filtered_data as bvp_y
    @property
    def bvp_y(self):
        """Backward compatibility property for accessing filtered data."""
        return self.filtered_data

    def _clean(self, data):
        """
        Clean the filtered data (placeholder for future processing).

        Args:
            data: Input signal array.

        Returns:
            Cleaned signal array.
        """
        return data

    def _normalize(self, data):
        """
        Apply min-max normalization to scale data to [0, 1] range.

        Args:
            data: Input signal array.

        Returns:
            Normalized signal array with values in [0, 1].
        """
        dmax = max(data)
        dmin = min(data)

        for i in range(0, len(data) - 1):
            data[i] = (data[i] - dmin) / (dmax - dmin)

        return data

    @staticmethod
    def _average_filter(data, size=5):
        """
        Apply a centered moving average filter with mirrored boundaries.

        Args:
            data: Input signal array.
            size (int): Window size for averaging. Will be rounded up to
                nearest even number.

        Returns:
            Smoothed signal array.
        """
        if size % 2 == 1:
            size += 1
        half = int(size / 2)
        length = len(data)
        avg = [0] * length

        # Mirror the head and tail to handle boundaries
        head = data[0:half]
        tail = data[-1 * half:]
        extended_data = np.concatenate([head[::-1], data, tail[::-1]])

        for i in range(0, length):
            avg[i] = np.sum(extended_data[i:i + size]) / size

        return avg

    @staticmethod
    def _rolling_avg_filter(data, size=5):
        """
        Apply a rolling average subtraction filter to remove baseline drift.

        This subtracts the moving average from the original signal, effectively
        acting as a high-pass filter that removes slow-varying components.

        Args:
            data: Input signal array.
            size (int): Window size for the moving average.

        Returns:
            Filtered signal with baseline removed.
        """
        avg = SignalFilter._average_filter(data, size)
        return [x[0] - x[1] for x in zip(data, avg)]

    @staticmethod
    def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        """
        Apply a Butterworth bandpass filter.

        Args:
            data: Input signal array.
            lowcut (float): Low cutoff frequency in Hz.
            highcut (float): High cutoff frequency in Hz.
            fs (int): Sampling frequency in Hz.
            order (int): Filter order.

        Returns:
            Bandpass filtered signal.
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')
        y = lfilter(b, a, data)
        return y

