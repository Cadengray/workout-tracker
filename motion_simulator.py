import numpy as np
import pandas as pd
import os

ACTIVITIES = {
    'A': 'Walking', 'B': 'Jogging', 'C': 'Stairs', 'D': 'Sitting',
    'E': 'Standing', 'F': 'Typing', 'G': 'Teeth brushing', 'H': 'Soup eating',
    'I': 'Chips eating', 'J': 'Pasta eating', 'K': 'Drinking',
    'L': 'Sandwich eating', 'M': 'Kicking', 'O': 'Catch', 'P': 'Dribbling',
    'Q': 'Writing', 'R': 'Clapping', 'S': 'Folding'
}


def load_wisdm(filepath, activity_filter=None, user_filter=None):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"WISDM file not found at '{filepath}'.\n"
            "Download from: https://www.cis.fordham.edu/wisdm/dataset.php"
        )

    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 6:
                continue
            try:
                user_id   = int(parts[0].strip())
                activity  = parts[1].strip()
                timestamp = int(parts[2].strip())
                x = float(parts[3].strip())
                y = float(parts[4].strip())
                z = float(parts[5].strip().rstrip(';'))
                rows.append((user_id, activity, timestamp, x, y, z))
            except ValueError:
                continue

    df = pd.DataFrame(rows, columns=['user_id', 'activity', 'timestamp', 'x', 'y', 'z'])

    if activity_filter is not None:
        if isinstance(activity_filter, str):
            activity_filter = [activity_filter]
        df = df[df['activity'].isin(activity_filter)]

    if user_filter is not None:
        if isinstance(user_filter, int):
            user_filter = [user_filter]
        df = df[df['user_id'].isin(user_filter)]

    if df.empty:
        raise ValueError("No data matched your filters. Check activity/user codes.")

    # sort by timestamp and remove large gaps (separate recording sessions)
    df = df.sort_values('timestamp').reset_index(drop=True)
    gaps = df['timestamp'].diff().fillna(0)
    median_gap = gaps[gaps > 0].median()
    df = df[gaps < median_gap * 10].reset_index(drop=True)

    df['activity_name'] = df['activity'].map(ACTIVITIES).fillna(df['activity'])

    # build time axis from sample index + estimated sample rate
    # this is robust to WISDM's inconsistent timestamp epochs
    median_interval_ns = df['timestamp'].diff().median()
    fs = 1e9 / median_interval_ns
    df['t'] = df.index / fs

    return df, fs


def get_magnitude(df):
    df = df.copy()
    df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    return df


def generate_motion():
    print("[motion_simulator] No WISDM file loaded — using simulated data.")
    t = np.linspace(0, 20, 200)
    motion = np.sqrt(3) * (np.sin(t) + np.random.normal(0, 0.15, len(t)))
    return t, motion, 10.0
