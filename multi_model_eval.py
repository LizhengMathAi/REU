from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image

from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import Wav2Vec2Model, AutoImageProcessor, AutoModel

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# Config
# ============================================================

AUDIO_DIR = Path("audio")          # contains .flv/.wav/.mp4/.avi/.m4a
OPENFACE_DIR = Path("openface")    # contains .csv and *_aligned folders

AUDIO_CKPT = Path("audio_wav2vec2.pt")
CSV_CKPT = Path("csv_mlp.pt")
VISION_CKPT = Path("vision_vit.pt")

CLASS_NAMES = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
AUDIO_EXTS = ["*.flv", "*.wav", "*.mp4", "*.avi", "*.m4a"]
IMAGE_EXTS = ["*.bmp", "*.jpg", "*.jpeg", "*.png"]

TEST_SIZE = 0.2
SEED = 42
BATCH_SIZE = 8
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default ensemble weights. Tune these on validation set, not test set.
W_AUDIO = 0.25
W_CSV = 0.50
W_VISION = 0.25


# ============================================================
# Reproducibility / filename parsing
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def crema_stem(path):
    """Return CREMA-D base id, e.g. 1001_DFA_HAP_XX."""
    stem = Path(path).stem
    if stem.endswith("_aligned"):
        stem = stem[:-len("_aligned")]
    return stem


def parse_stem(stem):
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Bad CREMA-D stem: {stem}")
    actor = int(parts[0])
    label = CLASS_NAMES.index(parts[2]) if parts[2] in CLASS_NAMES else -1
    return actor, label


# ============================================================
# Collect modality files
# ============================================================

def collect_audio_files(audio_dir):
    items = {}
    for pattern in AUDIO_EXTS:
        for p in sorted(Path(audio_dir).glob(pattern)):
            stem = crema_stem(p)
            try:
                actor, label = parse_stem(stem)
                if label >= 0:
                    items[stem] = {"path": p, "actor": actor, "label": label}
            except Exception as e:
                print("Skip audio:", p.name, "|", e)
    return items


def collect_csv_files(openface_dir):
    items = {}
    for p in sorted(Path(openface_dir).glob("*.csv")):
        stem = crema_stem(p)
        try:
            actor, label = parse_stem(stem)
            if label >= 0:
                items[stem] = {"path": p, "actor": actor, "label": label}
        except Exception as e:
            print("Skip csv:", p.name, "|", e)
    return items


def folder_has_images(folder):
    for ext in IMAGE_EXTS:
        if any(Path(folder).glob(ext)):
            return True
    return False


def collect_aligned_folders(openface_dir):
    items = {}
    for folder in sorted(Path(openface_dir).glob("*_aligned")):
        if not folder.is_dir() or not folder_has_images(folder):
            continue
        stem = crema_stem(folder)
        try:
            actor, label = parse_stem(stem)
            if label >= 0:
                items[stem] = {"path": folder, "actor": actor, "label": label}
        except Exception as e:
            print("Skip aligned:", folder.name, "|", e)
    return items


# ============================================================
# Audio model / dataset
# ============================================================

def load_audio_mono_16k(audio_path, target_sr=16000, max_seconds=5.0):
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    waveform = waveform.squeeze(0)

    max_len = int(target_sr * max_seconds)
    if waveform.numel() > max_len:
        waveform = waveform[:max_len]
    elif waveform.numel() < max_len:
        waveform = torch.nn.functional.pad(waveform, (0, max_len - waveform.numel()))
    return waveform.float()


class AudioDataset(Dataset):
    def __init__(self, stems, audio_items, sample_rate, max_seconds):
        self.stems = list(stems)
        self.audio_items = audio_items
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        item = self.audio_items[stem]
        waveform = load_audio_mono_16k(item["path"], self.sample_rate, self.max_seconds)
        y = torch.tensor(item["label"], dtype=torch.long)
        return stem, waveform, y


class Wav2Vec2EmotionClassifier(nn.Module):
    def __init__(self, pretrained_model="facebook/wav2vec2-base", num_classes=6, dropout=0.3, freeze_encoder=True):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model)
        hidden_size = self.wav2vec2.config.hidden_size
        if freeze_encoder:
            for p in self.wav2vec2.parameters():
                p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)


# ============================================================
# CSV model / dataset
# ============================================================

def get_feature_columns(df, feature_set="aur+pose+gaze"):
    df.columns = df.columns.str.strip()
    cols = []
    for fs in feature_set.split("+"):
        if fs == "aur":
            cols += [c for c in df.columns if c.startswith("AU") and c.endswith("_r")]
        elif fs == "auc":
            cols += [c for c in df.columns if c.startswith("AU") and c.endswith("_c")]
        elif fs == "pose":
            cols += [c for c in df.columns if c in ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]]
        elif fs == "gaze":
            cols += [c for c in df.columns if c in ["gaze_angle_x", "gaze_angle_y"]]
        else:
            raise ValueError(f"Unknown feature group: {fs}")
    return list(dict.fromkeys(cols))


def normalize_temporal_length(x, n_frames=30):
    if len(x) == 1:
        return np.repeat(x, n_frames, axis=0)
    t_old = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, n_frames)
    interp = interp1d(t_old, x, axis=0, kind="linear", fill_value="extrapolate")
    return interp(t_new)


def temporal_fft_features(x, n_freq=5, use_magnitude=True, remove_mean=True):
    if remove_mean:
        x = x - x.mean(axis=0, keepdims=True)
    Xf = np.fft.rfft(x, axis=0)
    n_freq = min(n_freq, Xf.shape[0])
    Xf = Xf[:n_freq]
    if use_magnitude:
        return np.abs(Xf)
    return np.concatenate([Xf.real, Xf.imag], axis=0)


def extract_csv_feature(csv_path, feature_set, n_frames, conf_th, n_freq=None):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if "success" in df.columns:
        df = df[df["success"] == 1]
    if "confidence" in df.columns:
        df = df[df["confidence"] > conf_th]
    cols = get_feature_columns(df, feature_set)
    if len(cols) == 0:
        raise ValueError("No valid feature columns found.")
    x = df[cols].astype(np.float32).values
    if len(x) == 0:
        raise ValueError("No valid frames after filtering.")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = normalize_temporal_length(x, n_frames=n_frames)
    if n_freq is not None:
        x = temporal_fft_features(x, n_freq=n_freq, use_magnitude=True, remove_mean=True)
    return x.reshape(-1).astype(np.float32)


class CSVDataset(Dataset):
    def __init__(self, stems, csv_items, feature_set, n_frames, conf_th, n_freq, scaler_mean, scaler_scale):
        self.stems = list(stems)
        self.csv_items = csv_items
        self.feature_set = feature_set
        self.n_frames = n_frames
        self.conf_th = conf_th
        self.n_freq = n_freq
        self.scaler_mean = scaler_mean.astype(np.float32)
        self.scaler_scale = scaler_scale.astype(np.float32)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        item = self.csv_items[stem]
        x = extract_csv_feature(item["path"], self.feature_set, self.n_frames, self.conf_th, self.n_freq)
        x = (x - self.scaler_mean) / np.maximum(self.scaler_scale, 1e-12)
        y = torch.tensor(item["label"], dtype=torch.long)
        return stem, torch.from_numpy(x.astype(np.float32)), y


class CSVOnlyMLP(nn.Module):
    def __init__(self, input_dim, num_classes=6):
        super().__init__()
        self.cls_head = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.cls_head(x)


# ============================================================
# Vision model / dataset
# ============================================================

def list_images(folder):
    files = []
    for ext in IMAGE_EXTS:
        files.extend(Path(folder).glob(ext))
    return sorted(files)


def sample_frame_paths(frame_paths, num_frames, random_sample=False):
    frame_paths = list(frame_paths)
    T = len(frame_paths)
    if T == 0:
        raise ValueError("No frames found.")
    if T >= num_frames:
        if random_sample:
            idx = sorted(np.random.choice(T, size=num_frames, replace=False).tolist())
        else:
            idx = np.linspace(0, T - 1, num_frames).round().astype(int).tolist()
    else:
        idx = np.linspace(0, T - 1, num_frames).round().astype(int).tolist()
    return [frame_paths[i] for i in idx]


def make_image_grid(frame_paths, grid_n=8, frame_size=112, out_size=384):
    num_frames = grid_n * grid_n
    selected = sample_frame_paths(frame_paths, num_frames=num_frames, random_sample=False)
    canvas = Image.new("RGB", (grid_n * frame_size, grid_n * frame_size))
    for k, img_path in enumerate(selected):
        img = Image.open(img_path).convert("RGB")
        img = img.resize((frame_size, frame_size), Image.BILINEAR)
        row = k // grid_n
        col = k % grid_n
        canvas.paste(img, (col * frame_size, row * frame_size))
    if out_size is not None:
        canvas = canvas.resize((out_size, out_size), Image.BILINEAR)
    return canvas


class VisionDataset(Dataset):
    def __init__(self, stems, vision_items, image_processor, grid_n, frame_size, grid_image_size):
        self.stems = list(stems)
        self.vision_items = vision_items
        self.processor = image_processor
        self.grid_n = grid_n
        self.frame_size = frame_size
        self.grid_image_size = grid_image_size
        self.frame_lists = {s: list_images(vision_items[s]["path"]) for s in self.stems}

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        item = self.vision_items[stem]
        grid_img = make_image_grid(
            self.frame_lists[stem],
            grid_n=self.grid_n,
            frame_size=self.frame_size,
            out_size=self.grid_image_size,
        )
        encoded = self.processor(images=grid_img, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)
        y = torch.tensor(item["label"], dtype=torch.long)
        return stem, pixel_values, y


class FrozenViTGridClassifier(nn.Module):
    def __init__(self, model_name, num_classes=6, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values):
        out = self.encoder(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0]
        return self.classifier(cls)


# ============================================================
# Prediction helpers
# ============================================================

@torch.no_grad()
def predict_probs(model, loader, input_kind):
    model.eval()
    probs_by_stem = {}
    y_by_stem = {}

    for batch in loader:
        stems = batch[0]
        x = batch[1].to(DEVICE)
        y = batch[2]

        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for s, p, yy in zip(stems, probs, y.numpy().tolist()):
            probs_by_stem[s] = p
            y_by_stem[s] = yy

    return probs_by_stem, y_by_stem


def evaluate_from_probs(name, stems, probs_by_stem, y_by_stem):
    y_true = np.array([y_by_stem[s] for s in stems])
    y_pred = np.array([np.argmax(probs_by_stem[s]) for s in stems])
    print(f"\n===== {name} =====")
    print("Samples:", len(stems))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    return y_true, y_pred


# ============================================================
# Main ensemble evaluation
# ============================================================

def main():
    set_seed(SEED)
    print("Device:", DEVICE)

    audio_items = collect_audio_files(AUDIO_DIR)
    csv_items = collect_csv_files(OPENFACE_DIR)
    vision_items = collect_aligned_folders(OPENFACE_DIR)

    print("Audio files:", len(audio_items))
    print("CSV files:", len(csv_items))
    print("Aligned folders:", len(vision_items))

    if len(audio_items) == 0:
        raise RuntimeError("No audio files found.")

    # Shared split from full audio actor set. This preserves your note:
    # audio has whole dataset; CSV/Vision may be incomplete.
    all_audio_actors = np.array(sorted({v["actor"] for v in audio_items.values()}))
    train_actors, test_actors = train_test_split(
        all_audio_actors,
        test_size=TEST_SIZE,
        random_state=SEED,
    )
    test_actor_set = set(test_actors.tolist())

    audio_test_stems = sorted([s for s, v in audio_items.items() if v["actor"] in test_actor_set])
    intersection_test_stems = sorted([
        s for s in audio_test_stems
        if s in csv_items and s in vision_items
    ])

    print("Full audio test samples:", len(audio_test_stems))
    print("Intersection test samples:", len(intersection_test_stems))

    # -------------------------
    # Load audio model
    # -------------------------
    audio_ckpt = torch.load(AUDIO_CKPT, map_location=DEVICE, weights_only=False)
    audio_model = Wav2Vec2EmotionClassifier(
        pretrained_model=audio_ckpt.get("pretrained_model", "facebook/wav2vec2-base"),
        num_classes=len(CLASS_NAMES),
        dropout=0.3,
        freeze_encoder=audio_ckpt.get("freeze_wav2vec2", True),
    ).to(DEVICE)
    audio_model.load_state_dict(audio_ckpt["model_state_dict"])

    sample_rate = audio_ckpt.get("sample_rate", 16000)
    max_seconds = audio_ckpt.get("max_seconds", 5.0)

    # -------------------------
    # Load CSV model
    # -------------------------
    csv_ckpt = torch.load(CSV_CKPT, map_location=DEVICE, weights_only=False)
    csv_model = CSVOnlyMLP(
        input_dim=csv_ckpt["input_dim"],
        num_classes=len(CLASS_NAMES),
    ).to(DEVICE)
    csv_model.load_state_dict(csv_ckpt["model_state_dict"])

    # -------------------------
    # Load vision model
    # -------------------------
    vision_ckpt = torch.load(VISION_CKPT, map_location=DEVICE, weights_only=False)
    model_name = vision_ckpt.get("model_name", "google/vit-large-patch16-384")
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    vision_model = FrozenViTGridClassifier(
        model_name=model_name,
        num_classes=len(CLASS_NAMES),
        dropout=0.3,
    ).to(DEVICE)
    vision_model.load_state_dict(vision_ckpt["model_state_dict"])

    # -------------------------
    # Predict: audio full test
    # -------------------------
    audio_full_loader = DataLoader(
        AudioDataset(audio_test_stems, audio_items, sample_rate, max_seconds),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    audio_full_probs, audio_full_y = predict_probs(audio_model, audio_full_loader, "audio")
    evaluate_from_probs("Audio-only on full audio test set", audio_test_stems, audio_full_probs, audio_full_y)

    # -------------------------
    # Predict: intersection test
    # -------------------------
    audio_loader = DataLoader(
        AudioDataset(intersection_test_stems, audio_items, sample_rate, max_seconds),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    csv_loader = DataLoader(
        CSVDataset(
            intersection_test_stems,
            csv_items,
            feature_set=csv_ckpt["feature_set"],
            n_frames=csv_ckpt["n_frames"],
            conf_th=csv_ckpt["conf_th"],
            n_freq=csv_ckpt["n_freq"],
            scaler_mean=np.asarray(csv_ckpt["scaler_mean"]),
            scaler_scale=np.asarray(csv_ckpt["scaler_scale"]),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    vision_loader = DataLoader(
        VisionDataset(
            intersection_test_stems,
            vision_items,
            image_processor=image_processor,
            grid_n=vision_ckpt["grid_n"],
            frame_size=vision_ckpt["frame_size"],
            grid_image_size=vision_ckpt["grid_image_size"],
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    audio_probs, y_audio = predict_probs(audio_model, audio_loader, "audio")
    csv_probs, y_csv = predict_probs(csv_model, csv_loader, "csv")
    vision_probs, y_vision = predict_probs(vision_model, vision_loader, "vision")

    evaluate_from_probs("Audio-only on intersection test set", intersection_test_stems, audio_probs, y_audio)
    evaluate_from_probs("CSV-only on intersection test set", intersection_test_stems, csv_probs, y_csv)
    evaluate_from_probs("Vision-only on intersection test set", intersection_test_stems, vision_probs, y_vision)

    # Sanity check: labels should agree across modalities.
    for s in intersection_test_stems:
        if not (y_audio[s] == y_csv[s] == y_vision[s]):
            raise RuntimeError(f"Label mismatch for {s}: audio={y_audio[s]}, csv={y_csv[s]}, vision={y_vision[s]}")

    ens_probs = {}
    ens_y = {}
    total_w = W_AUDIO + W_CSV + W_VISION
    for s in intersection_test_stems:
        ens_probs[s] = (
            W_AUDIO * audio_probs[s]
            + W_CSV * csv_probs[s]
            + W_VISION * vision_probs[s]
        ) / total_w
        ens_y[s] = y_audio[s]

    evaluate_from_probs(
        f"Weighted ensemble on intersection test set, weights=({W_AUDIO}, {W_CSV}, {W_VISION})",
        intersection_test_stems,
        ens_probs,
        ens_y,
    )


if __name__ == "__main__":
    main()
