import numpy as np
from motion_simulator import load_wisdm, get_magnitude, generate_motion
from filters import low_pass_filter, compute_jerk
from rep_counter import find_rep_indices
from graphs import plot_motion, plot_axes, plot_rep_comparison

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

WISDM_FILE    = 'WISDM_ar_v1.1_raw.txt'
ACTIVITY      = 'Jogging'
USER_ID       = 33
USE_REAL_DATA = True

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

if USE_REAL_DATA:
    try:
        df, fs = load_wisdm(WISDM_FILE, activity_filter=ACTIVITY, user_filter=USER_ID)
        df = get_magnitude(df)

        t      = df['t'].values
        motion = df['magnitude'].values

        print(f"Loaded {len(df)} samples")
        print(f"Activity: {df['activity_name'].iloc[0]}")
        print(f"Duration: {t[-1]:.1f} seconds")
        print(f"Sampling rate: ~{fs:.0f} Hz")

        plot_axes(df, max_seconds=20)

    except (FileNotFoundError, ValueError) as e:
        print(f"\nCould not load real data: {e}")
        print("Falling back to simulated data.\n")
        t, motion, fs = generate_motion()
else:
    t, motion, fs = generate_motion()

# ─────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────

filtered = low_pass_filter(motion, cutoff=0.8, fs=fs)
print(f"\nFiltering applied — cutoff: 3.0 Hz")

# ─────────────────────────────────────────────
# COUNT REPS ON FILTERED SIGNAL
# ─────────────────────────────────────────────

reps, rep_indices = find_rep_indices(filtered, fs=fs, min_rep_duration=0.8)
print(f"Detected reps: {reps}")

# ─────────────────────────────────────────────
# VISUALISE
# ─────────────────────────────────────────────

# show raw vs filtered with rep markers
plot_motion(t, motion, filtered=filtered, rep_indices=rep_indices)
