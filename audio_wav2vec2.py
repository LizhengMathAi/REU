from pathlib import Path
import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# =========================
# Config
# =========================

AUDIO_DIR = Path("audio")  # folder containing .flv/.wav/.mp4 files
AUDIO_EXTS = ["*.flv", "*.wav", "*.mp4", "*.avi", "*.m4a"]
CLASS_NAMES = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]

PRETRAINED_MODEL = "facebook/wav2vec2-base"
SAMPLE_RATE = 16000
MAX_SECONDS = 5.0              # CREMA-D clips are short; pad/truncate to this length
FREEZE_WAV2VEC2 = True         # first baseline: frozen encoder + trainable MLP head

BATCH_SIZE = 24                # Wav2Vec2 is memory-heavy; increase if GPU allows
EPOCHS = 30
LR = 1e-3                      # for classifier only when encoder is frozen
ENCODER_LR = 1e-5              # used only if FREEZE_WAV2VEC2=False
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
TEST_SIZE = 0.2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2


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

def parse_filename(audio_path):
    """
    CREMA-D example filename:
        1025_DFA_SAD_XX.flv
        actor_sentence_emotion_intensity.ext

    parts[0] = actor id
    parts[2] = emotion label
    """
    parts = Path(audio_path).stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {Path(audio_path).name}")
    actor = int(parts[0])
    label = CLASS_NAMES.index(parts[2]) if parts[2] in CLASS_NAMES else -1
    return actor, label


def collect_valid_files(audio_dir):
    audio_files = []
    for pattern in AUDIO_EXTS:
        audio_files.extend(sorted(Path(audio_dir).glob(pattern)))

    files, labels, actors = [], [], []
    for audio_path in audio_files:
        try:
            actor, label = parse_filename(audio_path)
            if label < 0:
                continue
            files.append(audio_path)
            labels.append(label)
            actors.append(actor)
        except Exception as e:
            print("Skip filename:", audio_path.name, "|", e)

    return np.array(files), np.array(labels), np.array(actors)


def load_audio_mono_16k(audio_path, target_sr=16000, max_seconds=5.0):
    """
    Return waveform tensor with shape [num_samples], sampled at target_sr.
    Pads/truncates to max_seconds for batching.
    """
    waveform, sr = torchaudio.load(str(audio_path))  # [channels, samples]

    # mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    waveform = waveform.squeeze(0)  # [samples]

    max_len = int(target_sr * max_seconds)
    if waveform.numel() > max_len:
        waveform = waveform[:max_len]
    elif waveform.numel() < max_len:
        pad_len = max_len - waveform.numel()
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))

    return waveform.float()


# =========================
# Dataset
# =========================

class CREMDAudioDataset(Dataset):
    """
    Dataloader samples audio/video files.
    Each item returns:
        waveform: FloatTensor, shape [num_samples]
        label: LongTensor scalar
    """
    def __init__(self, audio_files, labels, sample_rate=16000, max_seconds=5.0):
        self.audio_files = list(audio_files)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform = load_audio_mono_16k(
            audio_path,
            target_sr=self.sample_rate,
            max_seconds=self.max_seconds,
        )
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return waveform, y


# =========================
# Model
# =========================

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
        """
        input_values: [batch_size, num_samples]
        attention_mask: [batch_size, num_samples], optional
        """
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state  # [B, T_audio, H]

        # Mean pooling over Wav2Vec2 time steps.
        # Because we pad/truncate to fixed length, simple mean pooling is acceptable.
        pooled = hidden.mean(dim=1)         # [B, H]
        logits = self.classifier(pooled)    # [B, num_classes]
        return logits


# =========================
# Train / Eval
# =========================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for i, (waveform, y) in enumerate(loader):
        # waveform shape: [batch_size, num_samples]
        # y shape: [batch_size]
        waveform = waveform.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(waveform)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * waveform.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += waveform.size(0)

        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Training batch {i+1}/{len(loader)} | "
              f"batch loss {loss.item():.4f} | "
              f"batch acc {(pred == y).float().mean().item():.4f} | "
              f"running acc {correct / total:.4f}"
              , end="\r")
    print()  # newline after progress

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_y, all_pred = [], []

    for i, (waveform, y) in enumerate(loader):
        waveform = waveform.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(waveform)
        loss = criterion(logits, y)

        total_loss += loss.item() * waveform.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += waveform.size(0)

        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())

        print(f"Evaluating batch {i+1}/{len(loader)} | "
              f"batch loss {loss.item():.4f} | "
              f"batch acc {(pred == y).float().mean().item():.4f}", end="\r")
    print()  # newline after progress

    return total_loss / total, correct / total, np.array(all_y), np.array(all_pred)


def main():
    set_seed(SEED)
    print("Device:", DEVICE)

    audio_files, labels, actors = collect_valid_files(AUDIO_DIR)
    print("Number of audio/video files:", len(audio_files))
    print("Number of actors:", len(np.unique(actors)))

    if len(audio_files) == 0:
        raise RuntimeError(f"No valid audio/video files found in {AUDIO_DIR}")

    unique_actors = np.unique(actors)
    train_actors, test_actors = train_test_split(
        unique_actors,
        test_size=TEST_SIZE,
        random_state=SEED,
    )
    print("Number of training actors:", len(train_actors))
    print("Number of testing actors:", len(test_actors))

    train_mask = np.isin(actors, train_actors)
    test_mask = np.isin(actors, test_actors)

    train_files, train_labels = audio_files[train_mask], labels[train_mask]
    test_files, test_labels = audio_files[test_mask], labels[test_mask]
    print("Training samples:", len(train_files))
    print("Testing samples:", len(test_files))

    train_ds = CREMDAudioDataset(train_files, train_labels, SAMPLE_RATE, MAX_SECONDS)
    test_ds = CREMDAudioDataset(test_files, test_labels, SAMPLE_RATE, MAX_SECONDS)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    first_x, first_y = next(iter(train_loader))
    print("Input waveform shape per batch:", first_x.shape)  # [B, SAMPLE_RATE * MAX_SECONDS]
    print("Label shape per batch:", first_y.shape)

    class_count = np.bincount(train_labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    class_weight = class_count.sum() / np.maximum(class_count, 1.0)
    class_weight = class_weight / class_weight.mean()
    class_weight = torch.tensor(class_weight, dtype=torch.float32, device=DEVICE)

    model = Wav2Vec2EmotionClassifier(
        pretrained_model=PRETRAINED_MODEL,
        num_classes=len(CLASS_NAMES),
        dropout=DROPOUT,
        freeze_encoder=FREEZE_WAV2VEC2,
    ).to(DEVICE)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss(weight=class_weight)

    if FREEZE_WAV2VEC2:
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.wav2vec2.parameters(), "lr": ENCODER_LR},
                {"params": model.classifier.parameters(), "lr": LR},
            ],
            weight_decay=WEIGHT_DECAY,
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    best_acc = 0.0
    best_state = None

    print("\nStarting training...")
    for epoch in range(1, EPOCHS + 1):
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
            "class_names": CLASS_NAMES,
            "pretrained_model": PRETRAINED_MODEL,
            "sample_rate": SAMPLE_RATE,
            "max_seconds": MAX_SECONDS,
            "freeze_wav2vec2": FREEZE_WAV2VEC2,
        },
        "audio_wav2vec2.pt",
    )
    print("Saved checkpoint: audio_wav2vec2.pt")


if __name__ == "__main__":
    main()
