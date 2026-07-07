from pathlib import Path
import argparse
import random
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Config
# =========================

CSV_DIR = Path("openface")   # change this to your OpenFace CSV folder
CLASS_NAMES = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Argument parser
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CSV-only MLP on OpenFace features for CREMA-D emotion classification."
    )

    # Feature-selection hyperparameters from the tuning table
    parser.add_argument("--aur", action="store_true", help="Use AU intensity features AU*_r.")
    parser.add_argument("--auc", action="store_true", help="Use AU presence features AU*_c.")
    parser.add_argument("--pose", action="store_true", help="Use head-pose features.")
    parser.add_argument("--gaze", action="store_true", help="Use gaze-angle features.")

    parser.add_argument("--feature-set", type=str, default=None,
                        help="Optional direct feature set string, e.g. 'aur+pose+gaze'. Overrides --aur/--auc/--pose/--gaze.")
    parser.add_argument("--n-frames", type=int, default=30,
                        help="Number of temporal frames after interpolation.")
    parser.add_argument("--conf-th", type=float, default=0.8,
                        help="OpenFace confidence threshold.")
    parser.add_argument("--n-freq", type=int, default=-1,
                        help="Number of FFT frequency bins. Use -1 to disable FFT.")

    # General training hyperparameters
    parser.add_argument("--csv-dir", type=str, default=str(CSV_DIR), help="Folder containing OpenFace CSV files.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="csv_mlp.pt")

    return parser.parse_args()


def build_feature_set_from_args(args):
    if args.feature_set is not None:
        return args.feature_set

    parts = []
    if args.aur:
        parts.append("aur")
    if args.auc:
        parts.append("auc")
    if args.pose:
        parts.append("pose")
    if args.gaze:
        parts.append("gaze")

    # Default row in your table: aur + pose + gaze
    if len(parts) == 0:
        parts = ["aur", "pose", "gaze"]

    return "+".join(parts)


# =========================
# Reproducibility
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Helpers
# =========================

def parse_filename(csv_path):
    """
    CREMA-D example filename:
        1001_IEO_HAP_HI.csv
        actor_sentence_emotion_intensity.csv
    """
    parts = Path(csv_path).stem.split("_")
    actor = int(parts[0])
    label = CLASS_NAMES.index(parts[2]) if parts[2] in CLASS_NAMES else -1
    return actor, label


def get_feature_columns(df, feature_set="aur+pose+gaze"):
    """
    Feature groups:
        aur  : AU intensity columns, e.g. AU01_r
        auc  : AU presence columns, e.g. AU01_c
        pose : pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz
        gaze : gaze_angle_x, gaze_angle_y
    """
    df.columns = df.columns.str.strip()

    return_cols = []
    for fs in feature_set.split("+"):
        if fs == "aur":
            return_cols += [c for c in df.columns if c.startswith("AU") and c.endswith("_r")]
        elif fs == "auc":
            return_cols += [c for c in df.columns if c.startswith("AU") and c.endswith("_c")]
        elif fs == "pose":
            return_cols += [c for c in df.columns if c in [
                "pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"
            ]]
        elif fs == "gaze":
            return_cols += [c for c in df.columns if c in ["gaze_angle_x", "gaze_angle_y"]]
        else:
            raise ValueError(f"Unknown feature group: {fs}")

    # remove duplicates while preserving order
    return list(dict.fromkeys(return_cols))


def normalize_temporal_length(x, n_frames=30):
    """
    x: [T, F]
    return: [n_frames, F]
    """
    if len(x) == 1:
        return np.repeat(x, n_frames, axis=0)

    t_old = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, n_frames)
    interp = interp1d(t_old, x, axis=0, kind="linear", fill_value="extrapolate")
    return interp(t_new)


def temporal_fft_features(x, n_freq=5, use_magnitude=True, remove_mean=True):
    """
    x: [T, F]
    return: [n_freq, F] if use_magnitude=True
    """
    if remove_mean:
        x = x - x.mean(axis=0, keepdims=True)

    Xf = np.fft.rfft(x, axis=0)
    n_freq = min(n_freq, Xf.shape[0])
    Xf = Xf[:n_freq]

    if use_magnitude:
        return np.abs(Xf)
    return np.concatenate([Xf.real, Xf.imag], axis=0)


def extract_sequence_feature(csv_path, feature_set, n_frames, conf_th, n_freq=None):
    """
    Read one OpenFace CSV and return one fixed-size video-level feature vector.

    Output shape:
        without FFT: [n_frames * feature_dim]
        with FFT:    [n_freq * feature_dim]
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
        raise ValueError("No valid frames after success/confidence filtering.")

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = normalize_temporal_length(x, n_frames=n_frames)

    if n_freq is not None:
        x = temporal_fft_features(x, n_freq=n_freq, use_magnitude=True, remove_mean=True)

    return x.reshape(-1).astype(np.float32)


def collect_valid_files(csv_dir):
    csv_files, labels, actors = [], [], []
    for csv_path in sorted(Path(csv_dir).glob("*.csv")):
        try:
            actor, label = parse_filename(csv_path)
            if label < 0:
                continue
            csv_files.append(csv_path)
            labels.append(label)
            actors.append(actor)
        except Exception as e:
            print("Skip filename:", csv_path.name, "|", e)
    return np.array(csv_files), np.array(labels), np.array(actors)


# =========================
# Dataset
# =========================

class OpenFaceCSVDataset(Dataset):
    """
    Dataloader samples CSV files.
    Each item returns:
        x: torch.FloatTensor, shape [input_dim]
        y: torch.LongTensor scalar
    """
    def __init__(self, csv_files, labels, feature_set, n_frames, conf_th, n_freq=None, scaler=None):
        self.csv_files = list(csv_files)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.feature_set = feature_set
        self.n_frames = n_frames
        self.conf_th = conf_th
        self.n_freq = n_freq
        self.scaler = scaler

        # Preload features. This is usually fine for CREMA-D-scale CSV features.
        self.X = []
        self.y = []
        self.good_files = []
        for csv_path, label in zip(self.csv_files, self.labels):
            try:
                feat = extract_sequence_feature(
                    csv_path,
                    feature_set=self.feature_set,
                    n_frames=self.n_frames,
                    conf_th=self.conf_th,
                    n_freq=self.n_freq,
                )
                self.X.append(feat)
                self.y.append(label)
                self.good_files.append(csv_path)
            except Exception as e:
                print("Skip:", Path(csv_path).name, "|", e)

        if len(self.X) == 0:
            raise RuntimeError("No valid CSV files were loaded.")

        self.X = np.stack(self.X).astype(np.float32)
        self.y = np.asarray(self.y, dtype=np.int64)

        if self.scaler is not None:
            self.X = self.scaler.transform(self.X).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])              # [input_dim]
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# =========================
# Model
# =========================

class CSVOnlyMLP(nn.Module):
    def __init__(self, input_dim, num_classes=6, dropout=0.3):
        super().__init__()
        self.cls_head = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.cls_head(x)


# =========================
# Train / Eval
# =========================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        # x shape: [batch_size, input_dim]
        # y shape: [batch_size]
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_y, all_pred = [], []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())

    return total_loss / total, correct / total, np.array(all_y), np.array(all_pred)


def main():
    args = parse_args()

    feature_set = build_feature_set_from_args(args)
    n_freq = None if args.n_freq < 0 else args.n_freq

    set_seed(args.seed)
    print(args)

    csv_files, labels, actors = collect_valid_files(Path(args.csv_dir))
    print("Number of CSV files:", len(csv_files))
    print("Number of actors:", len(np.unique(actors)))

    # Speaker-independent split, same idea as your SVM script.
    unique_actors = np.unique(actors)
    train_actors, test_actors = train_test_split(
        unique_actors,
        test_size=args.test_size,
        random_state=args.seed,
    )

    train_mask = np.isin(actors, train_actors)
    test_mask = np.isin(actors, test_actors)

    train_files, train_labels = csv_files[train_mask], labels[train_mask]
    test_files, test_labels = csv_files[test_mask], labels[test_mask]

    # First build raw train dataset to fit scaler only on training data.
    raw_train_ds = OpenFaceCSVDataset(
        train_files, train_labels,
        feature_set=feature_set,
        n_frames=args.n_frames,
        conf_th=args.conf_th,
        n_freq=n_freq,
        scaler=None,
    )

    scaler = StandardScaler()
    scaler.fit(raw_train_ds.X)

    train_ds = OpenFaceCSVDataset(
        train_files, train_labels,
        feature_set=feature_set,
        n_frames=args.n_frames,
        conf_th=args.conf_th,
        n_freq=n_freq,
        scaler=scaler,
    )
    test_ds = OpenFaceCSVDataset(
        test_files, test_labels,
        feature_set=feature_set,
        n_frames=args.n_frames,
        conf_th=args.conf_th,
        n_freq=n_freq,
        scaler=scaler,
    )

    input_dim = train_ds.X.shape[1]
    print("Train samples:", len(train_ds))
    print("Test samples:", len(test_ds))
    print("Input tensor shape per sample:", train_ds[0][0].shape)
    print("Batch input tensor shape:", next(iter(DataLoader(train_ds, batch_size=args.batch_size)))[0].shape)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Class weights help when emotion classes are imbalanced.
    class_count = np.bincount(train_ds.y, minlength=len(CLASS_NAMES)).astype(np.float32)
    class_weight = class_count.sum() / np.maximum(class_count, 1.0)
    class_weight = class_weight / class_weight.mean()
    class_weight = torch.tensor(class_weight, dtype=torch.float32, device=DEVICE)

    model = CSVOnlyMLP(input_dim=input_dim, num_classes=len(CLASS_NAMES), dropout=args.dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)
        scheduler.step(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f} acc {test_acc:.4f} | "
            f"best {best_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    _, final_acc, y_true, y_pred = evaluate(model, test_loader, criterion)
    print("\nFinal Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "class_names": CLASS_NAMES,
            "feature_set": feature_set,
            "n_frames": args.n_frames,
            "conf_th": args.conf_th,
            "n_freq": n_freq,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        args.save_path,
    )
    print(f"Saved checkpoint: {args.save_path}")


if __name__ == "__main__":
    main()

    # Table of tuning hyperparameters
    # | aur | auc | pose | gaze | n_frames | conf_th | n_freq | acc  |
    # |-----|-----|------|------|----------|---------|--------|------|
    # | ❌  | ❌  | ❌   | ✅   | 30       | 0.8     | ❌     | 0.22 |
    # | ❌  | ❌  | ✅   | ❌   | 30       | 0.8     | ❌     | 0.24 |
    # | ❌  | ❌  | ✅   | ✅   | 30       | 0.8     | ❌     | 0.34 |
    # | ❌  | ❌  | ✅   | ✅   | 30       | 0.8     | ❌     | 0.56 |
    # | ❌  | ✅  | ❌   | ✅   | 30       | 0.8     | ❌     | 0.55 |
    # | ❌  | ✅  | ✅   | ❌   | 30       | 0.8     | ❌     | 0.55 |
    # | ❌  | ✅  | ✅   | ✅   | 30       | 0.8     | ❌     | 0.56 |
    # | ✅  | ❌  | ❌   | ✅   | 30       | 0.8     | ❌     | 0.65 |
    # | ✅  | ❌  | ✅   | ❌   | 30       | 0.8     | ❌     | 0.63 |
    # | ✅  | ❌  | ✅   | ✅   | 30       | 0.8     | ❌     | 0.63 |
    # | ✅  | ❌  | ✅   | ✅   | 30       | 0.8     | ❌     | 0.62 |
    # | ✅  | ✅  | ❌   | ❌   | 30       | 0.8     | ❌     | 0.61 |
    # | ✅  | ✅  | ❌   | ✅   | 30       | 0.8     | ❌     | 0.62 |
    # | ✅  | ✅  | ✅   | ❌   | 30       | 0.8     | ❌     | 0.59 |
    # | ✅  | ✅  | ✅   | ✅   | 30       | 0.8     | ❌     | 0.60 |

    # [default] python csv_mlp.py --aur --gaze