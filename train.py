import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
from model import PunchClassifier, StanceClassifier
# ── Which dataset to train on ───────────────────────────────
# "frontal"  -> training_data.json          -> punch_classifier_best.pt
# "sideways" -> sideways_training_data.json -> punch_classifier_sideways_best.pt
user_input = input("Enter '0' to train frontal punches or '1' to train sideways punches or '2' to train stance classification: ").strip().lower()
if user_input == "0":
    MODE = "frontal"
elif user_input == "1":
    MODE = "sideways"
elif user_input == "2":
    MODE = "stance"
elif user_input == "3":
    MODE = "all"
else:
    raise ValueError("Invalid input. Please enter '0', '1', '2', or '3'.")

assert MODE in ("frontal", "sideways", "stance", "all"), f"Unknown MODE: {MODE}"
if MODE == "frontal":
    DATA_FILE          = 'training_data.json'
    LABEL_ENCODER_FILE = 'label_encoder.pkl'
    MODEL_BEST_FILE     = 'punch_classifier_best.pt'
    MODEL_FINAL_FILE    = 'punch_classifier.pt'
    SUFFIX = "_frontal"
    DATA_KEY = "frames"  # punch snapshots store a 30-frame sequence of features
    MODEL = PunchClassifier
elif MODE == "sideways":
    DATA_FILE          = 'sideways_training_data.json'
    LABEL_ENCODER_FILE = 'label_encoder_sideways.pkl'
    MODEL_BEST_FILE     = 'punch_classifier_sideways_best.pt'
    MODEL_FINAL_FILE    = 'punch_classifier_sideways.pt'
    DATA_KEY = "frames"  # punch snapshots store a 30-frame sequence of features
    SUFFIX = "_sideways"
    MODEL = PunchClassifier
else: 
    DATA_FILE          = 'stance_data.json'
    LABEL_ENCODER_FILE = 'label_encoder_stance.pkl'
    MODEL_BEST_FILE     = 'stance_classifier_best.pt'
    MODEL_FINAL_FILE    = 'stance_classifier.pt'
    SUFFIX = "_stance"
    DATA_KEY = "features"  # stance snapshots store a flat feature vector
    MODEL = StanceClassifier

print(f"[MODE] Training on: {DATA_FILE}")

# ── Feature names (must match extract_features() order in get_values.py) ──
if MODE in ("frontal", "sideways"):
    FEATURE_NAMES = [
        "elbowAngleL",            # 0
        "elbowAngleR",            # 1
        "shoulderAngleL",         # 2
        "shoulderAngleR",         # 3
        "wristHeightL",           # 4
        "wristHeightR",           # 5
        "wristLextension_horz",   # 6
        "wristRextension_horz",   # 7
        "wristLextension_forward",# 8
        "wristRextension_forward",# 9
        "shoulderTilt",           # 10
        "hip_tilt",               # 11
        "hipAngleL",              # 12
        "hipAngleR",              # 13
        "legAngleL",              # 14
        "legAngleR",              # 15
        "wristAboveElbowL",       # 16
        "wristAboveElbowR",       # 17
        "elbowHeightL",           # 18
        "elbowHeightR",           # 19
        "wristLateralL",          # 20
        "wristLateralR",          # 21
        "elbowFlareL",            # 22
        "elbowFlareR",            # 23
        "wrist_foot_extensionL",  # 24
        "wrist_foot_extensionR",  # 25
    ]
elif MODE == "stance":
    FEATURE_NAMES = [
        "foot_x_diff",              # 0
        "foot_z_diff",              # 1
        "shoulder_x_diff",          # 2
        "shoulder_z_diff",          # 3
        "wristLextension_forward",  # 4
        "wristRextension_forward",  # 5
        "hip_x_diff",               # 6
        "direction",                # 7
        "foot_x_diff_canon",        # 8
        "shoulder_z_diff_canon",    # 9
        "hip_z_diff_canon",         # 10
    ]
# ── Load data ──────────────────────────────────────────────
with open(DATA_FILE) as f:
    raw = json.load(f)

X_np = np.array([d[DATA_KEY] for d in raw], dtype=np.float32)  # (N, 30, features) or (N, features)
labels = [d['label'] for d in raw]

num_features_in_data = X_np.shape[2] if DATA_KEY == "frames" else X_np.shape[1]
if num_features_in_data != len(FEATURE_NAMES):
    print(f"[WARN] {DATA_FILE} has {num_features_in_data} features per frame, "
          f"but FEATURE_NAMES has {len(FEATURE_NAMES)} entries. "
          f"Check that extract_features() and FEATURE_NAMES are in sync before trusting the importance ranking below.")

le = LabelEncoder()
y_np = le.fit_transform(labels)
print("Classes:", le.classes_)
print("Samples per class:", np.bincount(y_np))

with open(LABEL_ENCODER_FILE, 'wb') as f:
    pickle.dump(le, f)

# ── Train/val split ────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
)

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val,   dtype=torch.long))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32)

num_features = X_np.shape[2] if DATA_KEY == "frames" else X_np.shape[1]
num_classes  = len(le.classes_)
model = MODEL(num_features, num_classes=num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.CrossEntropyLoss()

# ── Training loop with early stopping ─────────────────────
train_losses, val_losses, val_accs = [], [], []
best_val_acc      = 0.0
epochs_no_improve = 0
PATIENCE          = 15

for epoch in range(120):
    # Train
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validate
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch)
            val_loss += loss_fn(logits, y_batch).item()
            correct  += (logits.argmax(1) == y_batch).sum().item()

    train_losses.append(total_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    val_accs.append(correct / len(val_ds))

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | train loss: {train_losses[-1]:.4f} | "
              f"val loss: {val_losses[-1]:.4f} | val acc: {val_accs[-1]:.1%}")

    # Save best, early stop
    if val_accs[-1] > best_val_acc:
        best_val_acc = val_accs[-1]
        torch.save(model.state_dict(), MODEL_BEST_FILE)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} — best val acc: {best_val_acc:.1%}")
            break

# Save final weights too
torch.save(model.state_dict(), MODEL_FINAL_FILE)
print(f"\nBest val acc: {best_val_acc:.1%}  →  {MODEL_BEST_FILE}")
print(f"Final weights →  {MODEL_FINAL_FILE}")

# # ── Plots ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 4))

# Loss curves
axes[0].plot(train_losses, label='train')
axes[0].plot(val_losses,   label='val')
axes[0].set_title('Loss')
axes[0].legend()

# Val accuracy
axes[1].plot(val_accs)
axes[1].set_title('Val Accuracy')
axes[1].set_ylim(0, 1)

sub_file = None
if MODE == "frontal":
    sub_file = f"punch"
elif MODE == "sideways":
    sub_file = f"sideways"
elif MODE == "stance":
    sub_file = f"stance"

OUTPUT_DIR = os.path.join("analysis_output", sub_file)
def out_path(filename):
    return os.path.join(OUTPUT_DIR, filename)
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.tight_layout()
plt.savefig(out_path(f'training_curves{SUFFIX}.png'))
plt.show()
print("Plots saved to", out_path(f'training_curves{SUFFIX}.png'))
