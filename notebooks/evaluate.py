import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, ConfusionMatrixDisplay
)
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
IMAGE_ROOT     = os.path.join(BASE_DIR, "..", "data")
PROJECT_ROOT   = os.path.dirname(BASE_DIR)
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results", "evaluation")

DISEASE_COLS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
BATCH_SIZE   = 128

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────
class CheXpertDataset(Dataset):
    def __init__(self, df, image_root, transform=None):
        self.image_root = image_root
        self.transform  = transform
        self.paths      = df["Path"].values
        self.labels     = df[DISEASE_COLS].values.astype("float32")

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


# ── Full Evaluation ───────────────────────────────────────────────────────────
def evaluate(preds, labels, thresholds):
    print("\n── Full Test Evaluation ────────────────────────────────────────────")
    print(f"{'Disease':<22} {'AUC':>6}  {'F1':>6}  {'Precision':>10}  {'Recall':>8}")
    print("-" * 62)

    results = []
    for i, col in enumerate(DISEASE_COLS):
        thresh       = thresholds[col]
        binary_preds = (preds[:, i] >= thresh).astype(int)

        auc  = roc_auc_score(labels[:, i], preds[:, i])
        f1   = f1_score(labels[:, i], binary_preds, zero_division=0)
        prec = precision_score(labels[:, i], binary_preds, zero_division=0)
        rec  = recall_score(labels[:, i], binary_preds, zero_division=0)

        print(f"{col:<22} {auc:>6.4f}  {f1:>6.4f}  {prec:>10.4f}  {rec:>8.4f}")
        results.append({"Disease": col, "AUC": auc, "F1": f1,
                         "Precision": prec, "Recall": rec, "Threshold": thresh})

    mean_auc = np.mean([r["AUC"] for r in results])
    mean_f1  = np.mean([r["F1"]  for r in results])
    print("-" * 62)
    print(f"{'Mean':<22} {mean_auc:>6.4f}  {mean_f1:>6.4f}")

    return pd.DataFrame(results)


# ── Confusion Matrices ────────────────────────────────────────────────────────
def plot_confusion_matrices(preds, labels, thresholds):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for i, col in enumerate(DISEASE_COLS):
        thresh       = thresholds[col]
        binary_preds = (preds[:, i] >= thresh).astype(int)
        cm = confusion_matrix(labels[:, i], binary_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Neg", "Pos"])
        disp.plot(ax=axes[i], colorbar=False)
        axes[i].set_title(col, fontsize=10)
    plt.suptitle("Confusion Matrices — Test Set", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrices saved to {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Data ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "train.csv"))
    df = df.fillna(0).replace(-1.0, 1.0)
    df["Path"] = df["Path"].str.replace("CheXpert-v1.0-small/", "", regex=False)
    df["PatientID"] = df["Path"].apply(lambda x: x.split("/")[1])

    patients = df["PatientID"].unique().to_numpy()
    train_patients, temp_patients = train_test_split(patients, test_size=0.30, random_state=42)
    val_patients,   test_patients = train_test_split(temp_patients, test_size=0.50, random_state=42)

    val_data  = df[df["PatientID"].isin(val_patients)]
    test_data = df[df["PatientID"].isin(test_patients)]
    print(f"Val: {len(val_data)}, Test: {len(test_data)}")

    # ── Transform ─────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5330, 0.5330, 0.5330], std=[0.0349, 0.0349, 0.0349])
    ])

    val_loader = DataLoader(
        CheXpertDataset(val_data, IMAGE_ROOT, transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        CheXpertDataset(test_data, IMAGE_ROOT, transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # ── Load Model ────────────────────────────────────────────────────────────
    device = torch.device("cuda")
    model  = get_model().to(device)
    ckpt   = torch.load(
        os.path.join(CHECKPOINT_DIR, "densenet121_full_final.pth"),
        map_location=device, weights_only=False
    )
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint — Val AUC at save time: {ckpt['val_auc']:.4f}")

    # ── Tune thresholds on VAL ─────────────────────────────────────────────────
    val_preds, val_labels = run_inference(model, val_loader, device)
    thresholds = {}
    from sklearn.metrics import roc_curve
    for i, col in enumerate(DISEASE_COLS):
        fpr, tpr, thresh = roc_curve(val_labels[:, i], val_preds[:, i])
        best_idx         = np.argmax(tpr - fpr)
        thresholds[col]  = float(thresh[best_idx])
    print("Thresholds (val-tuned):", thresholds)

    # ── Evaluate on TEST ───────────────────────────────────────────────────────
    test_preds, test_labels = run_inference(model, test_loader, device)
    results_df = evaluate(test_preds, test_labels, thresholds)
    plot_confusion_matrices(test_preds, test_labels, thresholds)

    # ── Save results ───────────────────────────────────────────────────────────
    save_path = os.path.join(RESULTS_DIR, "test_evaluation.csv")
    results_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
