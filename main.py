import numpy as np
from motion_simulator import load_wisdm, get_magnitude, generate_motion
from filters import low_pass_filter, compute_jerk
from rep_counter import find_rep_indices, segment_reps
from form_analyzer import analyze_reps, summarize, print_report
from exercise_classifier import train_classifier, predict_activity
from graphs import plot_motion, plot_axes, plot_rep_comparison

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

WISDM_FILE    = 'WISDM_ar_v1.1_raw.txt'
ACTIVITY      = 'Jogging'
USER_ID       = 33
USE_REAL_DATA = True
TRAIN_MODEL   = True   # set False after first run to skip retraining

# activities to train the classifier on
TRAIN_ACTIVITIES = ['Jogging', 'Walking', 'Sitting', 'Standing', 'Stairs']

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
# TRAIN CLASSIFIER (first run only)
# ─────────────────────────────────────────────

if TRAIN_MODEL:
    print("\nTraining exercise classifier...")
    data_by_activity = {}
    for activity in TRAIN_ACTIVITIES:
        try:
            act_df, _ = load_wisdm(WISDM_FILE, activity_filter=activity)
            act_df = get_magnitude(act_df)
            data_by_activity[activity] = act_df['magnitude'].values
            print(f"  Loaded {len(act_df)} samples for '{activity}'")
        except ValueError:
            print(f"  Skipping '{activity}' — no data found")

    if len(data_by_activity) >= 2:
        clf, le, report = train_classifier(data_by_activity, fs)
        print("\nClassifier accuracy report:")
        print(report)
    else:
        print("Not enough activities to train — skipping classifier.")

# ─────────────────────────────────────────────
# PREDICT ACTIVITY
# ─────────────────────────────────────────────

try:
    predicted, confidence = predict_activity(motion, fs)
    print(f"\nPredicted activity: {predicted} (confidence: {confidence:.0%})")
except FileNotFoundError:
    print("\nNo classifier model found — skipping prediction.")

# ─────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────

filtered = low_pass_filter(motion, cutoff=0.8, fs=fs)
print(f"\nFiltering applied — cutoff: 0.8 Hz")

# ─────────────────────────────────────────────
# COUNT REPS + SEGMENT
# ─────────────────────────────────────────────

reps, rep_indices = find_rep_indices(filtered, fs=fs, min_rep_duration=0.8)
segments = segment_reps(filtered, rep_indices)
print(f"Detected reps: {reps}")

# ─────────────────────────────────────────────
# FORM ANALYSIS
# ─────────────────────────────────────────────

rep_metrics = analyze_reps(filtered, rep_indices, segments, fs)
summary     = summarize(rep_metrics)
print_report(rep_metrics, summary)

# ─────────────────────────────────────────────
# VISUALISE
# ─────────────────────────────────────────────

plot_motion(t, motion, filtered=filtered, rep_indices=rep_indices)
plot_rep_comparison(segments, title="Rep-by-rep form comparison")
