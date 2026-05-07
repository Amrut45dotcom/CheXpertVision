import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_ROOT = os.path.join(BASE_DIR, "..", "data")
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

DISEASE_COLS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
BATCH_SIZE = 128


# ── Dataset ───────────────────────────────────────────────────────────────────
class CheXpertDataset(Dataset):
    def __init__(self, df, image_root, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.paths = df["Path"].values
        self.labels = df[DISEASE_COLS].values.astype("float32")

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


# ── Model ─────────────────────────────────────────────────────────────────────
def get_model():
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 5)
    )
    return model


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for images, labels in tqdm(loader, desc="Running inference"):
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ── Youden's J Threshold Tuning ───────────────────────────────────────────────
def find_optimal_thresholds(preds, labels):
    thresholds = {}
    for i, col in enumerate(DISEASE_COLS):
        fpr, tpr, thresh = roc_curve(labels[:, i], preds[:, i])
        j_scores = tpr - fpr  # Youden's J
        best_idx = np.argmax(j_scores)
        thresholds[col] = float(thresh[best_idx])
    return thresholds


# ── Metrics at Tuned Thresholds ───────────────────────────────────────────────
def evaluate_with_thresholds(preds, labels, thresholds):
    print("\n── Results with Tuned Thresholds ──────────────────────────────")
    print(f"{'Disease':<22} {'AUC':>6}  {'Threshold':>10}  {'Sensitivity':>12}  {'Specificity':>12}")
    print("-" * 70)
    aucs = []
    for i, col in enumerate(DISEASE_COLS):
        auc = roc_auc_score(labels[:, i], preds[:, i])
        thresh = thresholds[col]
        binary_preds = (preds[:, i] >= thresh).astype(int)
        tp = ((binary_preds == 1) & (labels[:, i] == 1)).sum()
        fn = ((binary_preds == 0) & (labels[:, i] == 1)).sum()
        tn = ((binary_preds == 0) & (labels[:, i] == 0)).sum()
        fp = ((binary_preds == 1) & (labels[:, i] == 0)).sum()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        aucs.append(auc)
        print(f"{col:<22} {auc:>6.4f}  {thresh:>10.4f}  {sensitivity:>12.4f}  {specificity:>12.4f}")
    print("-" * 70)
    print(f"{'Mean AUC':<22} {np.mean(aucs):>6.4f}")
    return thresholds


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Data ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "train.csv"))
    df = df.fillna(0).replace(-1.0, 1.0)
    df["Path"] = df["Path"].str.replace("CheXpert-v1.0-small/", "", regex=False)
    df["PatientID"] = df["Path"].apply(lambda x: x.split("/")[1])

    patients = df["PatientID"].unique()
    train_patients, temp_patients = train_test_split(patients, test_size=0.30, random_state=42)
    val_patients, test_patients   = train_test_split(temp_patients, test_size=0.50, random_state=42)
    test_data = df[df["PatientID"].isin(test_patients)]
    val_data = df[df["PatientID"].isin(val_patients)]
    print(f"Test set size: {len(test_data)}")

    # ── Transform ─────────────────────────────────────────────────────────────
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5330, 0.5330, 0.5330], std=[0.0349, 0.0349, 0.0349])
    ])

    test_loader = DataLoader(
        CheXpertDataset(test_data, IMAGE_ROOT, test_transform),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # ── Load Model ────────────────────────────────────────────────────────────
    device = torch.device("cuda")
    model = get_model().to(device)
    checkpoint = torch.load(
        os.path.join(CHECKPOINT_DIR, "densenet121_full_final.pth"),
        map_location=device,
	weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint — Val AUC at save time: {checkpoint['val_auc']:.4f}")


    # ── Tune thresholds on VAL set ────────────────────────────────────────────
    val_loader = DataLoader(
        CheXpertDataset(val_data, IMAGE_ROOT, test_transform),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    val_preds, val_labels = run_inference(model, val_loader, device)
    thresholds = find_optimal_thresholds(val_preds, val_labels)

# ── Evaluate on TEST set ──────────────────────────────────────────────────
    test_preds, test_labels = run_inference(model, test_loader, device)
    evaluate_with_thresholds(test_preds, test_labels, thresholds)


if __name__ == "__main__":
    main()
