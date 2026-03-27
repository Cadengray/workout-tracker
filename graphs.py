import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_motion(t, motion, title="Workout Motion Signal", filtered=None, rep_indices=None):
    """
    Plot the motion signal over time.

    Args:
        t: time array (seconds)
        motion: raw signal array (magnitude or single axis)
        title: plot title
        filtered: optional smoothed signal array to overlay
        rep_indices: optional list of indices where reps were detected
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(t, motion, color='#aaaaaa', linewidth=0.8, alpha=0.7, label='Raw signal')

    if filtered is not None:
        ax.plot(t, filtered, color='#534AB7', linewidth=1.5, label='Filtered signal')

    if rep_indices is not None and len(rep_indices) > 0:
        rep_times = [t[i] for i in rep_indices if i < len(t)]
        rep_vals  = [motion[i] for i in rep_indices if i < len(t)]
        ax.scatter(rep_times, rep_vals, color='#D85A30', zorder=5,
                   s=40, label=f'Reps detected ({len(rep_indices)})')

    ax.set_title(title, fontsize=13, fontweight='normal', pad=10)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Acceleration magnitude (m/s²)", fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_axes(df, max_seconds=20):
    """
    Plot x, y, z axes separately for a given window of data.
    Useful for understanding raw sensor orientation and noise.

    Args:
        df: DataFrame with columns t, x, y, z
        max_seconds: only plot up to this many seconds
    """
    df = df[df['t'] <= max_seconds]

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    colors = ['#D85A30', '#534AB7', '#1D9E75']
    labels = ['X axis', 'Y axis', 'Z axis']

    for i, (col, color, label) in enumerate(zip(['x', 'y', 'z'], colors, labels)):
        axes[i].plot(df['t'], df[col], color=color, linewidth=0.8)
        axes[i].set_ylabel(label, fontsize=10)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    fig.suptitle("Raw accelerometer axes", fontsize=13, fontweight='normal')
    plt.tight_layout()
    plt.show()


def plot_rep_comparison(rep_signals, title="Rep-by-rep comparison"):
    """
    Overlay each individual rep signal on the same plot.
    Each rep is normalized to the same length so they can be compared.
    This is the foundation of form analysis — spotting reps that look different.

    Args:
        rep_signals: list of 1D arrays, one per rep
        title: plot title
    """
    if not rep_signals:
        print("No reps to compare.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    n = len(rep_signals)
    cmap = plt.cm.get_cmap('cool', n)

    # normalize each rep to 100 points so they overlay cleanly
    normalized_x = np.linspace(0, 100, 100)

    for i, rep in enumerate(rep_signals):
        rep_x = np.linspace(0, 100, len(rep))
        rep_interp = np.interp(normalized_x, rep_x, rep)
        alpha = 0.4 if n > 5 else 0.7
        ax.plot(normalized_x, rep_interp, color=cmap(i), linewidth=1.2,
                alpha=alpha, label=f'Rep {i+1}' if n <= 8 else None)

    # overlay the mean rep
    all_interp = []
    for rep in rep_signals:
        rep_x = np.linspace(0, 100, len(rep))
        all_interp.append(np.interp(normalized_x, rep_x, rep))
    mean_rep = np.mean(all_interp, axis=0)
    ax.plot(normalized_x, mean_rep, color='black', linewidth=2,
            linestyle='--', label='Mean rep', zorder=5)

    ax.set_title(title, fontsize=13, fontweight='normal', pad=10)
    ax.set_xlabel("Rep progress (%)", fontsize=11)
    ax.set_ylabel("Acceleration magnitude", fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5)

    if n <= 8:
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()
