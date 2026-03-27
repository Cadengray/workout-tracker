import numpy as np
from scipy.signal import butter, filtfilt


def low_pass_filter(signal, cutoff=3.0, fs=10.0, order=4):
    """
    Smooth a noisy signal by removing high-frequency noise.

    For jogging at ~10Hz sampling rate, the actual stride frequency is
    around 1.5-3Hz. Setting cutoff to 3.0Hz keeps the real motion
    and removes everything faster (noise, micro-vibrations).

    Args:
        signal: 1D numpy array of acceleration magnitude
        cutoff: frequency in Hz above which to remove (default 3.0)
        fs: sampling rate of the signal in Hz
        order: filter sharpness — higher = sharper cutoff (default 4)

    Returns:
        Smoothed signal as a numpy array
    """
    nyquist = fs / 2.0
    normal_cutoff = cutoff / nyquist

    # clamp to valid range — cutoff must be below nyquist
    normal_cutoff = min(normal_cutoff, 0.99)

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


def normalize(signal):
    """
    Scale a signal to the range [0, 1].
    Useful for comparing reps of different intensities.
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val == min_val:
        return np.zeros_like(signal)
    return (signal - min_val) / (max_val - min_val)


def compute_jerk(signal, fs):
    """
    Compute jerk — the rate of change of acceleration.
    High jerk = sudden, jerky movement. Low jerk = smooth, controlled movement.
    This is one of the core metrics for form quality analysis in later weeks.

    Args:
        signal: 1D acceleration magnitude array
        fs: sampling rate in Hz

    Returns:
        jerk array (same length as signal, first value is 0)
    """
    jerk = np.diff(signal, prepend=signal[0]) * fs
    return np.abs(jerk)
