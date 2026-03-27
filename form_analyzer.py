import numpy as np
from filters import compute_jerk


def analyze_reps(signal, rep_indices, segments, fs):
    """
    Analyze each individual rep and return a list of per-rep metrics.

    Args:
        signal: full filtered 1D signal array
        rep_indices: list of peak indices from find_rep_indices
        segments: list of 1D arrays from segment_reps, one per rep
        fs: sampling rate in Hz

    Returns:
        List of dicts, one per rep, each containing:
            - rep_num, duration, peak_acceleration, smoothness,
              consistency, flags
    """
    if not segments:
        return []

    # filter out segments that are outliers in duration before analysis
    # this removes badly segmented reps that would skew all averages
    durations = np.array([len(seg) / fs for seg in segments])
    median_dur = np.median(durations)
    valid_mask = (durations > median_dur * 0.3) & (durations < median_dur * 3.0)
    segments   = [s for s, v in zip(segments, valid_mask) if v]

    if not segments:
        return []

    normalized  = _normalize_segments(segments)
    mean_rep    = np.mean(normalized, axis=0)

    durations       = [len(seg) / fs for seg in segments]
    mean_duration   = np.mean(durations)
    std_duration    = np.std(durations)

    smoothness_scores = [np.mean(compute_jerk(seg, fs)) for seg in segments]
    mean_smoothness   = np.mean(smoothness_scores)
    std_smoothness    = np.std(smoothness_scores)

    # set consistency threshold relative to actual data
    consistencies = []
    for norm_seg in normalized:
        corr = np.corrcoef(norm_seg, mean_rep)[0, 1]
        consistencies.append(max(0.0, corr))
    mean_consistency = np.mean(consistencies)
    # flag reps that are more than 1.5 std below the session mean consistency
    consistency_threshold = max(0.0, mean_consistency - 1.5 * np.std(consistencies))

    results = []
    for i, (seg, norm_seg, duration, smoothness, consistency) in enumerate(
        zip(segments, normalized, durations, smoothness_scores, consistencies)
    ):
        peak_accel = float(np.max(seg))
        flags = _check_flags(
            duration, mean_duration, std_duration,
            smoothness, mean_smoothness, std_smoothness,
            consistency, consistency_threshold
        )
        results.append({
            'rep_num':           i + 1,
            'duration':          round(duration, 3),
            'peak_acceleration': round(peak_accel, 3),
            'smoothness':        round(smoothness, 3),
            'consistency':       round(consistency, 3),
            'flags':             flags
        })

    return results


def _normalize_segments(segments, n_points=100):
    x_out = np.linspace(0, 1, n_points)
    normalized = []
    for seg in segments:
        if len(seg) < 2:
            normalized.append(np.zeros(n_points))
            continue
        x_in = np.linspace(0, 1, len(seg))
        normalized.append(np.interp(x_out, x_in, seg))
    return normalized


def _check_flags(duration, mean_dur, std_dur,
                 smoothness, mean_smooth, std_smooth,
                 consistency, consistency_threshold):
    flags = []

    if std_dur > 0:
        if duration < mean_dur - 1.5 * std_dur:
            pct = round((1 - duration / mean_dur) * 100)
            flags.append(f"Too fast — {pct}% quicker than your average")
        elif duration > mean_dur + 1.5 * std_dur:
            pct = round((duration / mean_dur - 1) * 100)
            flags.append(f"Too slow — {pct}% slower than your average")

    if std_smooth > 0 and smoothness > mean_smooth + 1.5 * std_smooth:
        flags.append("Choppy movement — loss of control detected")

    if consistency < consistency_threshold:
        flags.append("Inconsistent form — shape differs from your average rep")

    return flags


def summarize(rep_metrics):
    if not rep_metrics:
        return {}

    durations   = [r['duration'] for r in rep_metrics]
    smoothness  = [r['smoothness'] for r in rep_metrics]
    consistency = [r['consistency'] for r in rep_metrics]
    flagged     = [r['rep_num'] for r in rep_metrics if r['flags']]

    mid = len(rep_metrics) // 2
    first_half_smooth  = np.mean(smoothness[:mid]) if mid > 0 else 0
    second_half_smooth = np.mean(smoothness[mid:]) if mid > 0 else 0
    fatigue_detected   = second_half_smooth > first_half_smooth * 1.2

    return {
        'total_reps':       len(rep_metrics),
        'mean_duration':    round(np.mean(durations), 3),
        'std_duration':     round(np.std(durations), 3),
        'mean_smoothness':  round(np.mean(smoothness), 3),
        'mean_consistency': round(np.mean(consistency), 3),
        'flagged_reps':     flagged,
        'fatigue_detected': fatigue_detected
    }


def print_report(rep_metrics, summary):
    print("\n" + "="*50)
    print("FORM ANALYSIS REPORT")
    print("="*50)
    print(f"Total reps:        {summary['total_reps']}")
    print(f"Avg rep duration:  {summary['mean_duration']:.2f}s "
          f"(±{summary['std_duration']:.2f}s)")
    print(f"Avg smoothness:    {summary['mean_smoothness']:.2f} "
          f"(lower = smoother)")
    print(f"Avg consistency:   {summary['mean_consistency']:.2f} "
          f"(1.0 = perfect)")
    print(f"Fatigue detected:  {'Yes' if summary['fatigue_detected'] else 'No'}")

    if summary['flagged_reps']:
        print(f"\nFlagged reps: {summary['flagged_reps']}")
        print("\nDetails:")
        for r in rep_metrics:
            if r['flags']:
                print(f"  Rep {r['rep_num']:>3}: {' | '.join(r['flags'])}")
    else:
        print("\nNo form issues detected — great session!")
    print("="*50)
