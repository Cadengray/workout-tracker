import numpy as np


def find_rep_indices(signal, threshold=None, fs=10.0, min_rep_duration=0.5):
    """
    Detect reps in a motion signal using peak detection.

    Args:
        signal: 1D numpy array of acceleration magnitude
        threshold: minimum peak height. If None, auto-set to mean + 0.5 * std.
        fs: sampling rate in Hz — used to compute minimum gap between peaks
        min_rep_duration: minimum time in seconds between two reps (default 0.5s)
                          At jogging cadence (~170 steps/min), one stride = ~0.7s
                          so 0.5s gives us room without missing reps.

    Returns:
        (rep_count, list of peak indices)
    """
    if threshold is None:
        threshold = np.mean(signal) + 0.5 * np.std(signal)

    min_gap_samples = int(fs * min_rep_duration)

    peaks = []
    last_peak = -min_gap_samples

    for i in range(1, len(signal) - 1):
        is_peak = signal[i] > signal[i - 1] and signal[i] > signal[i + 1]
        above_threshold = signal[i] > threshold
        far_enough = (i - last_peak) >= min_gap_samples

        if is_peak and above_threshold and far_enough:
            peaks.append(i)
            last_peak = i

    return len(peaks), peaks


def count_reps(signal, threshold=None, fs=10.0):
    """
    Simple wrapper — returns just the rep count.
    """
    reps, _ = find_rep_indices(signal, threshold, fs)
    return reps


def segment_reps(signal, rep_indices, context_samples=5):
    """
    Slice the signal into individual rep segments.
    Each segment runs from the midpoint before a peak to the midpoint after it.
    This is the foundation of per-rep form analysis.

    Args:
        signal: full 1D signal array
        rep_indices: list of peak indices from find_rep_indices
        context_samples: extra samples to include either side of each segment

    Returns:
        List of 1D arrays, one per rep
    """
    if len(rep_indices) < 2:
        return [signal[rep_indices[0]]] if rep_indices else []

    segments = []
    for i, peak in enumerate(rep_indices):
        if i == 0:
            start = max(0, peak - (rep_indices[1] - peak) // 2)
        else:
            start = (rep_indices[i - 1] + peak) // 2

        if i == len(rep_indices) - 1:
            end = min(len(signal), peak + (peak - rep_indices[i - 1]) // 2)
        else:
            end = (peak + rep_indices[i + 1]) // 2

        start = max(0, start - context_samples)
        end   = min(len(signal), end + context_samples)
        segments.append(signal[start:end])

    return segments
