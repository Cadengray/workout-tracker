import numpy as np
import pickle
import os
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_features(window):
    """
    Extract a fixed-size feature vector from a signal window.
    These features capture the statistical and frequency properties
    of the motion — enough for a classifier to distinguish exercises.

    Args:
        window: 1D numpy array of acceleration magnitude values

    Returns:
        1D numpy array of features
    """
    if len(window) < 4:
        return np.zeros(13)

    # time domain features
    mean    = np.mean(window)
    std     = np.std(window)
    min_val = np.min(window)
    max_val = np.max(window)
    rng     = max_val - min_val
    skew    = float(stats.skew(window))
    kurt    = float(stats.kurtosis(window))

    # energy — sum of squared values, normalized by window length
    energy = np.sum(window ** 2) / len(window)

    # zero crossing rate — how often signal crosses its mean
    mean_centered = window - mean
    zero_crossings = np.sum(np.diff(np.sign(mean_centered)) != 0)
    zcr = zero_crossings / len(window)

    # jerk — mean rate of change (roughness of movement)
    jerk = np.mean(np.abs(np.diff(window)))

    # frequency domain — dominant frequency and its power
    fft_vals  = np.abs(np.fft.rfft(window))
    freqs     = np.fft.rfftfreq(len(window))
    dom_freq  = freqs[np.argmax(fft_vals[1:]) + 1] if len(fft_vals) > 1 else 0
    fft_power = np.sum(fft_vals ** 2) / len(fft_vals)

    # percentile spread — difference between 75th and 25th percentile
    iqr = float(np.percentile(window, 75) - np.percentile(window, 25))

    return np.array([
        mean, std, min_val, max_val, rng,
        skew, kurt, energy, zcr, jerk,
        dom_freq, fft_power, iqr
    ])


def extract_windows(signal, window_size, step_size):
    """
    Slide a window across the signal and extract features from each window.
    This converts a raw time series into a matrix of feature vectors
    that the classifier can be trained on.

    Args:
        signal: 1D numpy array
        window_size: number of samples per window
        step_size: number of samples to advance between windows

    Returns:
        2D array of shape (n_windows, n_features)
    """
    features = []
    for start in range(0, len(signal) - window_size, step_size):
        window = signal[start:start + window_size]
        features.append(extract_features(window))
    return np.array(features)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_classifier(data_by_activity, fs, window_seconds=2.0, step_seconds=1.0,
                     model_path='classifier.pkl'):
    """
    Train a Random Forest classifier on labeled activity data.

    Args:
        data_by_activity: dict mapping activity name (str) to 1D signal array
                          e.g. {'Jogging': array, 'Walking': array, ...}
        fs: sampling rate in Hz
        window_seconds: length of each feature window in seconds
        step_seconds: step between windows in seconds
        model_path: where to save the trained model

    Returns:
        Trained classifier, label encoder, and classification report string
    """
    window_size = int(window_seconds * fs)
    step_size   = int(step_seconds * fs)

    X, y = [], []
    for activity, signal in data_by_activity.items():
        features = extract_windows(signal, window_size, step_size)
        X.append(features)
        y.extend([activity] * len(features))

    X = np.vstack(X)
    y = np.array(y)

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 activities to train a classifier.")

    # encode string labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred  = clf.predict(X_test)
    report  = classification_report(y_test, y_pred, target_names=le.classes_)

    # save model and encoder together
    with open(model_path, 'wb') as f:
        pickle.dump({'classifier': clf, 'label_encoder': le, 'fs': fs}, f)

    print(f"Model saved to '{model_path}'")
    return clf, le, report


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

def load_classifier(model_path='classifier.pkl'):
    """
    Load a previously trained classifier from disk.

    Returns:
        (classifier, label_encoder, fs) tuple
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'.\n"
            "Run train_classifier() first."
        )
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['classifier'], data['label_encoder'], data['fs']


def predict_activity(signal, fs, model_path='classifier.pkl',
                     window_seconds=2.0, step_seconds=1.0):
    """
    Predict the exercise type for an unlabeled signal.
    Uses majority vote across all windows for robustness.

    Args:
        signal: 1D numpy array of acceleration magnitude
        fs: sampling rate in Hz
        model_path: path to saved model file
        window_seconds: window size (must match training)
        step_seconds: step size (must match training)

    Returns:
        (predicted_activity, confidence) where confidence is the
        fraction of windows that agreed on the top prediction
    """
    clf, le, _ = load_classifier(model_path)

    window_size = int(window_seconds * fs)
    step_size   = int(step_seconds * fs)

    features = extract_windows(signal, window_size, step_size)
    if len(features) == 0:
        return "Unknown", 0.0

    predictions = clf.predict(features)
    labels      = le.inverse_transform(predictions)

    # majority vote
    unique, counts = np.unique(labels, return_counts=True)
    top_idx    = np.argmax(counts)
    activity   = unique[top_idx]
    confidence = counts[top_idx] / len(labels)

    return activity, round(confidence, 3)
