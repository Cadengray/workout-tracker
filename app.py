import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from motion_simulator import load_wisdm, get_magnitude
from filters import low_pass_filter
from rep_counter import find_rep_indices, segment_reps
from form_analyzer import analyze_reps, summarize
from exercise_classifier import predict_activity, train_classifier

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Workout Analyzer",
    page_icon="🏃",
    layout="wide"
)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("🏃 Workout Form Analyzer")
st.markdown("Upload your smartwatch accelerometer data to get rep counts, form scores, and fatigue analysis.")

# ─────────────────────────────────────────────
# SIDEBAR — settings
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    filter_cutoff = st.slider(
        "Filter cutoff (Hz)",
        min_value=0.3, max_value=3.0, value=0.8, step=0.1,
        help="Lower = smoother signal. Increase if reps are being missed."
    )

    min_rep_duration = st.slider(
        "Min rep duration (seconds)",
        min_value=0.3, max_value=2.0, value=0.8, step=0.1,
        help="Minimum time between reps. Increase for slower exercises."
    )

    st.markdown("---")
    st.markdown("**Data source**")
    data_source = st.radio(
        "Choose input",
        ["Upload CSV file", "Use WISDM dataset"],
        label_visibility="collapsed"
    )

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

df = None
fs = 10.0
motion = None
t = None

if data_source == "Upload CSV file":
    st.markdown("### Upload your data")
    st.markdown("CSV should have columns: `timestamp, x, y, z` (acceleration in m/s²)")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            st.success(f"Loaded {len(raw)} rows")

            # try to find x, y, z columns flexibly
            col_map = {c.lower(): c for c in raw.columns}
            x = raw[col_map['x']].values
            y = raw[col_map['y']].values
            z = raw[col_map['z']].values
            motion = np.sqrt(x**2 + y**2 + z**2)

            if 'timestamp' in col_map:
                ts = raw[col_map['timestamp']].values
                ts = ts - ts[0]
                intervals = np.diff(ts)
                median_interval = np.median(intervals[intervals > 0])
                fs = 1e9 / median_interval if median_interval > 1e6 else 1.0 / (median_interval / 1000)
                fs = max(1.0, min(fs, 200.0))
                t = np.arange(len(motion)) / fs
            else:
                t = np.arange(len(motion)) / fs

        except Exception as e:
            st.error(f"Could not parse file: {e}")
            st.info("Make sure your CSV has columns named x, y, z")

elif data_source == "Use WISDM dataset":
    st.markdown("### WISDM dataset")

    wisdm_path = st.text_input("Path to WISDM file", value="WISDM_ar_v1.1_raw.txt")

    col1, col2 = st.columns(2)
    with col1:
        activity = st.selectbox("Activity", [
            "Jogging", "Walking", "Sitting", "Standing"
        ])
    with col2:
        user_id = st.number_input("User ID", min_value=1, max_value=51, value=33)

    if st.button("Load data"):
        if not os.path.exists(wisdm_path):
            st.error(f"File not found: {wisdm_path}")
        else:
            with st.spinner("Loading..."):
                try:
                    df, fs = load_wisdm(wisdm_path, activity_filter=activity, user_filter=int(user_id))
                    df = get_magnitude(df)
                    t = df['t'].values
                    motion = df['magnitude'].values
                    st.success(f"Loaded {len(df)} samples — {t[-1]:.1f}s at ~{fs:.0f} Hz")
                except Exception as e:
                    st.error(str(e))

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────

if motion is not None and t is not None and len(motion) > 20:

    st.markdown("---")

    # classify activity
    model_path = "classifier.pkl"
    if os.path.exists(model_path):
        predicted, confidence = predict_activity(motion, fs, model_path)
        st.markdown(f"### Detected activity: **{predicted}** ({confidence:.0%} confidence)")
    
    # filter
    filtered = low_pass_filter(motion, cutoff=filter_cutoff, fs=fs)

    # detect reps
    reps, rep_indices = find_rep_indices(filtered, fs=fs, min_rep_duration=min_rep_duration)
    segments = segment_reps(filtered, rep_indices)

    # form analysis
    rep_metrics = analyze_reps(filtered, rep_indices, segments, fs)
    summary = summarize(rep_metrics)

    # ── top metrics ──
    st.markdown("### Session summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total reps", summary.get('total_reps', 0))
    m2.metric("Avg duration", f"{summary.get('mean_duration', 0):.2f}s")
    m3.metric("Consistency", f"{summary.get('mean_consistency', 0):.2f}")
    m4.metric("Fatigue", "⚠️ Detected" if summary.get('fatigue_detected') else "✅ None")

    flagged = summary.get('flagged_reps', [])
    if flagged:
        st.warning(f"⚠️ {len(flagged)} reps flagged — see details below")
    else:
        st.success("✅ No form issues detected")

    # ── signal plot ──
    st.markdown("### Motion signal")
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, motion, color='#cccccc', linewidth=0.6, label='Raw')
    ax.plot(t, filtered, color='#534AB7', linewidth=1.5, label='Filtered')
    if rep_indices:
        rep_t = [t[i] for i in rep_indices if i < len(t)]
        rep_v = [motion[i] for i in rep_indices if i < len(t)]
        ax.scatter(rep_t, rep_v, color='#D85A30', s=30, zorder=5,
                   label=f'Reps ({reps})')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── rep comparison plot ──
    if segments and len(segments) > 1:
        st.markdown("### Rep-by-rep comparison")
        fig2, ax2 = plt.subplots(figsize=(12, 3))
        n = len(segments)
        cmap = plt.cm.get_cmap('cool', n)
        x_norm = np.linspace(0, 100, 100)
        all_interp = []
        for i, seg in enumerate(segments):
            xi = np.linspace(0, 100, len(seg))
            interp = np.interp(x_norm, xi, seg)
            all_interp.append(interp)
            ax2.plot(x_norm, interp, color=cmap(i), linewidth=0.8, alpha=0.4)
        mean_rep = np.mean(all_interp, axis=0)
        ax2.plot(x_norm, mean_rep, color='black', linewidth=2,
                 linestyle='--', label='Mean rep')
        ax2.set_xlabel("Rep progress (%)")
        ax2.set_ylabel("Acceleration magnitude")
        ax2.legend(fontsize=8)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── flagged reps detail ──
    if rep_metrics:
        st.markdown("### Per-rep details")
        rows = []
        for r in rep_metrics:
            rows.append({
                'Rep': r['rep_num'],
                'Duration (s)': r['duration'],
                'Smoothness': r['smoothness'],
                'Consistency': r['consistency'],
                'Flags': ' | '.join(r['flags']) if r['flags'] else '✅ Good'
            })
        rep_df = pd.DataFrame(rows)

        def highlight_flags(row):
            if row['Flags'] != '✅ Good':
                return ['background-color: #fff3cd'] * len(row)
            return [''] * len(row)

        st.dataframe(
            rep_df.style.apply(highlight_flags, axis=1),
            use_container_width=True,
            hide_index=True
        )

else:
    st.info("👈 Load some data using the options above to get started.")
