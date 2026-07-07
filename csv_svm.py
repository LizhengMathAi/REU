from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# =========================
# Config
# =========================

CSV_DIR = Path("openface")   # change this to your CSV folder
CLASS_NAMES = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
N_FRAMES = 30
CONF_TH = 0.8
N_FREQ = 5


# =========================
# Helpers
# =========================

def parse_filename(csv_path):
    """
    Example filename:
    1001_IEO_HAP_HI.csv
    actor_sentence_emotion_intensity.csv
    """
    parts = Path(csv_path).stem.split("_")
    actor = int(parts[0])
    label = CLASS_NAMES.index(parts[2]) if parts[2] in CLASS_NAMES else -1
    return actor, label


# ============================================================
# Feature sets
# ============================================================


def get_feature_columns(df, feature_set="au+pose+gaze"):
    """
    Return feature columns according to the selected feature set.

    feature_set:
        auc       : Action Units (c) only
        aur       : Action Units (r) only
        pose      : Head pose only
        gaze      : Eye gaze only
    """

    df.columns = df.columns.str.strip()

    return_cols = []
    for fs in feature_set.split("+"):
        if fs == "aur":
            return_cols += [c for c in df.columns if c.startswith("AU") and c.endswith("_r")]
        elif fs == "auc":
            return_cols += [c for c in df.columns if c.startswith("AU") and c.endswith("_c")]
        elif fs == "pose":
            return_cols += [c for c in df.columns if c in ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]]
        elif fs == "gaze":
            return_cols += [c for c in df.columns if c in ["gaze_angle_x", "gaze_angle_y"]]
    return return_cols


# ============================================================
# Feature extraction
# ============================================================


def temporal_fft_features(x, n_freq=10, use_magnitude=True, remove_mean=True):
    """
    x: [n_frames, n_features]

    Return:
        if use_magnitude=True:
            [n_freq, n_features]
        else:
            [2 * n_freq, n_features]  # real + imaginary
    """
    if remove_mean:
        x = x - x.mean(axis=0, keepdims=True)

    Xf = np.fft.rfft(x, axis=0)  # [n_frames // 2 + 1, n_features]

    n_freq = min(n_freq, Xf.shape[0])
    Xf = Xf[:n_freq]

    if use_magnitude:
        return np.abs(Xf)

    return np.concatenate([Xf.real, Xf.imag], axis=0)


def normalize_temporal_length(x, n_frames=30):
    """
    x: [T, F]
    Return: [n_frames, F]
    """
    if len(x) == 1:
        return np.repeat(x, n_frames, axis=0)

    t_old = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, n_frames)

    interp = interp1d(
        t_old,
        x,
        axis=0,
        kind="linear",
        fill_value="extrapolate",
    )
    return interp(t_new)


def extract_sequence_feature(
    csv_path,
    feature_set="baseline",
    n_frames=30,
    conf_th=0.8,
    n_freq=None,
):
    """
    temporal_method:
        "interp"   : linear interpolation, return time-domain features
        "resample" : scipy.signal.resample, return time-domain features
        "fft"      : linear interpolation first, then FFT low-frequency features
    """

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "success" in df.columns:
        df = df[df["success"] == 1]

    if "confidence" in df.columns:
        df = df[df["confidence"] > conf_th]

    feature_cols = get_feature_columns(df, feature_set)

    if len(feature_cols) == 0:
        raise ValueError("No valid feature columns found.")

    x = df[feature_cols].astype(np.float32).values

    if len(x) == 0:
        raise ValueError("No valid frames.")

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Resize the temporal dimension to fixed length
    x = normalize_temporal_length(x, n_frames=n_frames)  # [n_frames, n_features]

    # Apply Fast Fourier Transform
    if n_freq is not None:
        x = temporal_fft_features(
            x,
            n_freq=n_freq,
            use_magnitude=True,  # TODO:
            remove_mean=True,  # TODO:
        )  # [n_freq, n_features]

    return x.reshape(-1)


def build_dataset(csv_dir):
    X = []
    y = []
    actors = []
    files = []

    csv_files = sorted(Path(csv_dir).glob("*.csv"))

    for csv_path in csv_files:
        try:
            # +--------+---------------+------+------+------+---------+-----+
            # |        | aur+pose+gaze | aur  | pose | gaze | au_pose | all |
            # +--------+---------------+------+------+------+---------+-----+
            # | interp | 0.58          | 0.6  | 0.24 | 0.28 | 0.59    | 0.4 |
            # | fft    |               |      |      |      |         |     |
            # +--------+---------------+---0--+------+------+---------+-----+
            feat = extract_sequence_feature(
                csv_path,
                feature_set="aur+pose+gaze",
                n_frames=N_FRAMES,
                conf_th=CONF_TH,
                n_freq=None,
            )

            actor, label = parse_filename(csv_path)

            X.append(feat)
            y.append(label)
            actors.append(actor)
            files.append(csv_path.name)

        except Exception as e:
            print("Skip:", csv_path.name, "|", e)

    X = np.array(X)
    y = np.array(y)
    actors = np.array(actors)

    return X, y, actors, files


# =========================
# Build dataset
# =========================

X, y, actors, files = build_dataset(CSV_DIR)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of actors:", len(np.unique(actors)))
print("Number of files:", len(files))


# =========================
# Speaker-independent split
# =========================

unique_actors = np.unique(actors)

train_actors, test_actors = train_test_split(
    unique_actors,
    test_size=0.2,
    random_state=42
)

train_mask = np.isin(actors, train_actors)
test_mask = np.isin(actors, test_actors)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# =========================
# Train SVM
# =========================
clf = Pipeline([
    ("scaler", StandardScaler()),
    # ("pca", PCA(n_components=0.95)),
    ("svm", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
    )),
])

clf.fit(X_train, y_train)


# =========================
# Evaluate
# =========================

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print(
    classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES
    )
)

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
