import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import cv2

# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT  = os.path.expanduser("~/project_xray/Project_XRay/checkpoints/densenet121_full_final.pth")
THRESH_CSV  = os.path.expanduser("~/project_xray/Project_XRay/checkpoints/optimal_thresholds.csv")
OUTPUT_DIR  = os.path.expanduser("~/project_xray/Project_XRay/results/gradcam")
CLASSES     = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

IMAGE_PATHS = [
    "/data/b23_himanshu_shekhar/project_xray/Project_XRay/data/valid/patient64693/study1/view1_frontal.jpg",
    "/data/b23_himanshu_shekhar/project_xray/Project_XRay/data/valid/patient64559/study1/view1_frontal.jpg",
    "/data/b23_himanshu_shekhar/project_xray/Project_XRay/data/valid/patient64570/study1/view1_frontal.jpg",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Load thresholds ──────────────────────────────────────────────────────────
thresh_df  = pd.read_csv(THRESH_CSV, header=0)
THRESHOLDS = dict(zip(thresh_df.columns, thresh_df.iloc[0].values))
print("Thresholds loaded:", THRESHOLDS)

# ── Model ────────────────────────────────────────────────────────────────────
def build_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(1024, len(CLASSES))
    )
    return model

model = build_model()
ckpt  = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
state = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()
print("Model loaded.")

# ── Transform ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5330, 0.5330, 0.5330], std=[0.0349, 0.0349, 0.0349]),
])

# ── Grad-CAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)           # (1, 5)
        score  = output[0, class_idx]
        score.backward()

        grads = self.gradients[0]                   # (C, H, W)
        acts  = self.activations[0]                 # (C, H, W)
        weights = grads.mean(dim=(1, 2))            # (C,)
        cam = (weights[:, None, None] * acts).sum(0)
        cam = torch.clamp(cam, min=0).cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam = cv2.resize(cam, (224, 224))
        return cam, torch.sigmoid(output)[0].cpu().detach().numpy()

target_layer = model.features.denseblock4
gradcam      = GradCAM(model, target_layer)

# ── Process each image ───────────────────────────────────────────────────────
for img_path in IMAGE_PATHS:
    if not os.path.exists(img_path):
        print(f"[SKIP] Not found: {img_path}")
        continue

    pil_img    = Image.open(img_path).convert("RGB")
    img_resized = pil_img.resize((224, 224))
    inp        = transform(pil_img).unsqueeze(0).to(DEVICE)

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    axes[0].imshow(img_resized, cmap="gray")
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    for i, cls in enumerate(CLASSES):
        inp.requires_grad_(True)
        cam, probs = gradcam.generate(inp, i)

        detected = probs[i] >= THRESHOLDS.get(cls, 0.5)
        label    = f"{cls}\n{probs[i]:.3f} {'✓' if detected else ''}"

        img_np = np.array(img_resized).astype(np.float32) / 255.0
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        overlay = 0.5 * img_np + 0.5 * heatmap
        overlay = np.clip(overlay, 0, 1)

        axes[i+1].imshow(overlay)
        axes[i+1].set_title(label, fontsize=9,
                             color="green" if detected else "black")
        axes[i+1].axis("off")

    patient = img_path.split("/")[-3]
    out_path = os.path.join(OUTPUT_DIR, f"gradcam_{patient}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

print("Done.")

