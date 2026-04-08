import os
import time
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_ROOT = os.path.join(BASE_DIR, "..", "data")

# =========================
# Dataset
# =========================
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
# Training Functions
# =========================
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
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for images, labels in loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
    return total_loss / len(loader)

# =========================
# Main Pipeline
# =========================
def main():
    torch.backends.cudnn.benchmark = True

    df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "train.csv"))
    df = df.fillna(0).replace(-1.0, 1.0)
    df["Path"] = df["Path"].str.replace("CheXpert-v1.0-small/", "", regex=False)
    df["PatientID"] = df["Path"].apply(lambda x: x.split("/")[1])

    # DELETE the 3 stray lines that were here

    patients = df["PatientID"].unique()
    keep_patients, _ = train_test_split(patients, test_size=0.50, random_state=42)
    df = df[df["PatientID"].isin(keep_patients)]

    train_patients, temp_patients = train_test_split(keep_patients, test_size=0.30, random_state=42)
    val_patients, test_patients = train_test_split(temp_patients, test_size=0.50, random_state=42)

    train_data = df[df["PatientID"].isin(train_patients)]
    val_data = df[df["PatientID"].isin(val_patients)]
    test_data = df[df["PatientID"].isin(test_patients)]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5330, 0.5330, 0.5330], [0.0349, 0.0349, 0.0349])
    ])

    # Efficient Data Loading
    train_loader = DataLoader(
        CheXpertDataset(train_data, IMAGE_ROOT, transform),
        batch_size=64, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        CheXpertDataset(val_data, IMAGE_ROOT, transform),
        batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model Switch: EfficientNet-B0
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    
    device = torch.device("cuda")
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda') 

    # Training Loop
    epochs = 5
    for epoch in range(epochs):
        start_time = time.time()
        t_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        v_loss = validate(model, val_loader, criterion, device)
        
        duration = (time.time() - start_time) / 60
        print(f"Epoch {epoch+1} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Time: {duration:.2f} min")

        # Save Checkpoint
        torch.save(model.state_dict(), f"effnet_b0_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()