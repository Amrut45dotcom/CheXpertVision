import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_ROOT = os.path.join(BASE_DIR, "..", "data")
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


MODEL_NAME = "densenet121"
BATCH_SIZE = 32



# Dataset
class CheXpertDataset(Dataset):
    def __init__(self, df, image_root, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.paths = df["Path"].values
        self.labels = df[
            ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
        ].values.astype("float32")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_root, self.paths[idx].replace("/", os.sep))
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.labels[idx])

# =========================
# CHANGE 2: get_model factory
# =========================
def get_model(name):
    if name == "densenet121":
        model = models.densenet121(weights="IMAGENET1K_V1")
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 5)   # DenseNet: flat classifier
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 5)
    elif name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 5)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}")


# Training Functions
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for images, labels in loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    disease_cols = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    aucs = {}
    for i, col in enumerate(disease_cols):
        try:
            aucs[col] = roc_auc_score(all_labels[:, i], all_preds[:, i])
        except ValueError:
            aucs[col] = float('nan')
    mean_auc = np.nanmean(list(aucs.values()))
    return total_loss / len(loader), aucs, mean_auc

def plot_loss_curves(train_losses, val_losses, save_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    plt.plot(epochs, val_losses, 'r-o', label='Val Loss', linewidth=2, markersize=4)
    plt.title('Training vs Validation Loss', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Loss curve saved to {save_path}")


# Main Pipeline
def main():
    torch.backends.cudnn.benchmark = True
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "train.csv"))
    df = df.fillna(0).replace(-1.0, 1.0)
    df["Path"] = df["Path"].str.replace("CheXpert-v1.0-small/", "", regex=False)
    df["PatientID"] = df["Path"].apply(lambda x: x.split("/")[1])

    patients = df["PatientID"].unique()
    keep_patients, _ = train_test_split(patients, test_size=0.50, random_state=42)
    df = df[df["PatientID"].isin(keep_patients)]

    train_patients, temp_patients = train_test_split(keep_patients, test_size=0.30, random_state=42)
    val_patients, test_patients = train_test_split(temp_patients, test_size=0.50, random_state=42)

    train_data = df[df["PatientID"].isin(train_patients)]
    val_data   = df[df["PatientID"].isin(val_patients)]
    test_data  = df[df["PatientID"].isin(test_patients)]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomResizedCrop(size=224, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5330, 0.5330, 0.5330], std=[0.0349, 0.0349, 0.0349])
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5330, 0.5330, 0.5330], std=[0.0349, 0.0349, 0.0349])
    ])

    train_loader = DataLoader(
        CheXpertDataset(train_data, IMAGE_ROOT, train_transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        CheXpertDataset(val_data, IMAGE_ROOT, val_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )



    model = get_model(MODEL_NAME)
    print(f"\nModel: {MODEL_NAME}")
    count_params(model)


    device = torch.device("cuda")
    model = model.to(device)

    disease_cols = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    pos_weight = []
    for col in disease_cols:
        pos = (train_data[col] == 1).sum()
        neg = (train_data[col] == 0).sum()
        pos_weight.append(neg / pos)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    best_auc = 0
    epochs = 8
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        start_time = time.time()
        t_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        v_loss, aucs, mean_auc = validate(model, val_loader, criterion, device)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        duration = (time.time() - start_time) / 60

        print(f"Epoch {epoch+1} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Mean AUC: {mean_auc:.4f} | Time: {duration:.2f} min")
        for k, v in aucs.items():
            print(f"  {k}: {v:.4f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_auc,
            }, os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.pth"))
            print(f"  → Best model saved (AUC: {best_auc:.4f})")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': mean_auc,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_epoch_{epoch+1:02d}.pth"))
        

    plot_loss_curves(
        train_losses, val_losses,
        save_path=os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_loss_curve.png")
    )

if __name__ == "__main__":
    main()