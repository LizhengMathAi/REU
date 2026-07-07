from pathlib import Path
import random
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel


# =========================
# Config
# =========================

ALIGNED_ROOT = Path("openface")
CLASS_NAMES = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]

# Each video folder looks like:
# openface/1001_DFA_NEU_XX_aligned/frame_det_00_000001.bmp
IMAGE_EXTS = ["*.bmp", "*.jpg", "*.jpeg", "*.png"]

GRID_N = 8                  # sample GRID_N^2 frames; 8 -> 64 frames
FRAME_SIZE = 112            # original OpenFace aligned face size
GRID_IMAGE_SIZE = 384       # final grid image resized to ViT input size

# MODEL_NAME = "google/vit-base-patch16-384"
MODEL_NAME = "google/vit-large-patch16-384"
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
TEST_SIZE = 0.2
SEED = 42
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

def parse_folder_name(folder_path):
    """
    CREMA-D aligned folder example:
        1001_DFA_NEU_XX_aligned

    Original video name:
        actor_sentence_emotion_intensity.flv
    """
    stem = Path(folder_path).name
    if stem.endswith("_aligned"):
        stem = stem[:-len("_aligned")]

    parts = stem.split("_")
    actor = int(parts[0])
    label = CLASS_NAMES.index(parts[2]) if parts[2] in CLASS_NAMES else -1
    return actor, label


def collect_aligned_folders(aligned_root):
    folders, labels, actors = [], [], []
    aligned_root = Path(aligned_root)

    for folder in sorted(aligned_root.glob("*_aligned")):
        if not folder.is_dir():
            continue
        try:
            actor, label = parse_folder_name(folder)
            if label < 0:
                continue

            has_image = False
            for ext in IMAGE_EXTS:
                if any(folder.glob(ext)):
                    has_image = True
                    break
            if not has_image:
                print("Skip empty folder:", folder.name)
                continue

            folders.append(folder)
            labels.append(label)
            actors.append(actor)
        except Exception as e:
            print("Skip folder:", folder.name, "|", e)

    return np.array(folders), np.array(labels), np.array(actors)


def list_images(folder):
    files = []
    for ext in IMAGE_EXTS:
        files.extend(Path(folder).glob(ext))
    return sorted(files)


def sample_frame_paths(frame_paths, num_frames, random_sample=True):
    """
    Return exactly num_frames paths.
    If the video has fewer frames, repeat frames.
    """
    frame_paths = list(frame_paths)
    T = len(frame_paths)
    if T == 0:
        raise ValueError("No frames found.")

    if T >= num_frames:
        if random_sample:
            # sample sorted random indices to preserve temporal order
            idx = sorted(np.random.choice(T, size=num_frames, replace=False).tolist())
        else:
            idx = np.linspace(0, T - 1, num_frames).round().astype(int).tolist()
    else:
        idx = np.linspace(0, T - 1, num_frames).round().astype(int).tolist()

    return [frame_paths[i] for i in idx]


def make_image_grid(frame_paths, grid_n=4, frame_size=112, out_size=224):
    """
    Convert N^2 sampled aligned face frames into one image grid.

    Output:
        PIL RGB image, size [out_size, out_size]
    """
    num_frames = grid_n * grid_n
    selected = sample_frame_paths(frame_paths, num_frames=num_frames, random_sample=True)

    canvas = Image.new("RGB", (grid_n * frame_size, grid_n * frame_size))

    for k, img_path in enumerate(selected):
        img = Image.open(img_path).convert("RGB")
        img = img.resize((frame_size, frame_size), Image.BILINEAR)

        row = k // grid_n
        col = k % grid_n
        canvas.paste(img, (col * frame_size, row * frame_size))

    if out_size is not None:
        canvas = canvas.resize((out_size, out_size), Image.BILINEAR)

    # canvas.show() # uncomment for debugging and visualization

    return canvas


# =========================
# Dataset
# =========================

class AlignedGridDataset(Dataset):
    """
    Dataloader samples aligned-image folders.

    Each item returns:
        pixel_values: torch.FloatTensor, shape [3, 224, 224]
        y: torch.LongTensor scalar
    """
    def __init__(self, folders, labels, image_processor, grid_n=4, frame_size=112, out_size=224):
        self.folders = list(folders)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.processor = image_processor
        self.grid_n = grid_n
        self.frame_size = frame_size
        self.out_size = out_size

        self.frame_lists = []
        self.good_folders = []
        self.good_labels = []
        for folder, label in zip(self.folders, self.labels):
            frames = list_images(folder)
            if len(frames) == 0:
                print("Skip empty folder:", Path(folder).name)
                continue
            self.frame_lists.append(frames)
            self.good_folders.append(folder)
            self.good_labels.append(label)

        self.labels = np.asarray(self.good_labels, dtype=np.int64)
        if len(self.labels) == 0:
            raise RuntimeError("No valid aligned-image folders were loaded.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        grid_img = make_image_grid(
            self.frame_lists[idx],
            grid_n=self.grid_n,
            frame_size=self.frame_size,
            out_size=self.out_size,
        )

        encoded = self.processor(images=grid_img, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)   # [3, 224, 224]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return pixel_values, y


# =========================
# Model
# =========================

class FrozenViTGridClassifier(nn.Module):
    def __init__(self, model_name, num_classes=6, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        # Freeze ViT encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            # nn.LayerNorm(hidden_size),
            # nn.Dropout(dropout),
            # nn.Linear(hidden_size, num_classes),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values):
        # pixel_values: [B, 3, 224, 224]
        out = self.encoder(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0]  # [B, hidden_size]
        logits = self.classifier(cls)
        return logits


# =========================
# Train / Eval
# =========================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    # Keep frozen encoder in eval mode so dropout etc. stay disabled there.
    model.encoder.eval()

    total_loss, correct, total = 0.0, 0, 0

    for i, (pixel_values, y) in enumerate(loader):
        pixel_values = pixel_values.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(pixel_values)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * pixel_values.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += pixel_values.size(0)

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

    for pixel_values, y in loader:
        pixel_values = pixel_values.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(pixel_values)
        loss = criterion(logits, y)

        total_loss += loss.item() * pixel_values.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += pixel_values.size(0)

        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())

    return total_loss / total, correct / total, np.array(all_y), np.array(all_pred)


def main():
    set_seed(SEED)
    print("Device:", DEVICE)

    folders, labels, actors = collect_aligned_folders(ALIGNED_ROOT)
    print("Number of aligned folders:", len(folders))
    print("Number of actors:", len(np.unique(actors)))

    unique_actors = np.unique(actors)
    train_actors, test_actors = train_test_split(
        unique_actors,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    train_mask = np.isin(actors, train_actors)
    test_mask = np.isin(actors, test_actors)

    train_folders, train_labels = folders[train_mask], labels[train_mask]
    test_folders, test_labels = folders[test_mask], labels[test_mask]

    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    train_ds = AlignedGridDataset(
        train_folders,
        train_labels,
        image_processor=image_processor,
        grid_n=GRID_N,
        frame_size=FRAME_SIZE,
        out_size=GRID_IMAGE_SIZE,
    )
    test_ds = AlignedGridDataset(
        test_folders,
        test_labels,
        image_processor=image_processor,
        grid_n=GRID_N,
        frame_size=FRAME_SIZE,
        out_size=GRID_IMAGE_SIZE,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    first_x, first_y = train_ds[0]
    first_batch_x, first_batch_y = next(iter(train_loader))
    print("Input tensor shape per sample:", first_x.shape)
    print("Batch input tensor shape:", first_batch_x.shape)
    print("Label shape:", first_batch_y.shape)

    class_count = np.bincount(train_ds.labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    class_weight = class_count.sum() / np.maximum(class_count, 1.0)
    class_weight = class_weight / class_weight.mean()
    class_weight = torch.tensor(class_weight, dtype=torch.float32, device=DEVICE)

    model = FrozenViTGridClassifier(
        model_name=MODEL_NAME,
        num_classes=len(CLASS_NAMES),
        dropout=DROPOUT,
    ).to(DEVICE)

    # Only train classifier parameters because encoder is frozen.
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_acc = 0.0
    best_state = None

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
            "model_name": MODEL_NAME,
            "grid_n": GRID_N,
            "frame_size": FRAME_SIZE,
            "grid_image_size": GRID_IMAGE_SIZE,
        },
        "vision_vit.pt",
    )
    print("Saved checkpoint: vision_vit.pt")


if __name__ == "__main__":
    main()
