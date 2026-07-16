import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle

from model import StanceClassifier

# ── Feature names (must match get_stance_features() order in get_values.py) ──
FEATURE_NAMES = [
    "foot_x_diff",              # 0
    "foot_z_diff",              # 1
    "shoulder_x_diff",          # 2
    "shoulder_z_diff",          # 3
    "wristLextension_forward",  # 4
    "wristRextension_forward",  # 5
    "hip_x_diff",               # 6
    "direction",                # 7
]

# ── Load data ──────────────────────────────────────────────
with open('stance_data.json') as f:
    raw = json.load(f)

X_np = np.array([d['features'] for d in raw], dtype=np.float32)  # (N, features)
labels = [d['label'] for d in raw]

num_features_in_data = X_np.shape[1]
if num_features_in_data != len(FEATURE_NAMES):
    print(f"[WARN] stance_data.json has {num_features_in_data} features per sample, "
          f"but FEATURE_NAMES has {len(FEATURE_NAMES)} entries. "
          f"Check that get_stance_features() and FEATURE_NAMES are in sync before trusting the importance ranking below.")

le = LabelEncoder()
y_np = le.fit_transform(labels)
print("Classes:", le.classes_)
print("Samples per class:", np.bincount(y_np))

with open('stance_label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# ── Train/val split ────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
)

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val,   dtype=torch.long))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32)

# ── Model ──────────────────────────────────────────────────
num_features = X_np.shape[1]
num_classes  = len(le.classes_)
model = StanceClassifier(input_size=num_features, num_classes=num_classes)

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
        torch.save(model.state_dict(), 'stance_classifier_best.pt')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} — best val acc: {best_val_acc:.1%}")
            break

# Save final weights too
torch.save(model.state_dict(), 'stance_classifier.pt')
print(f"\nBest val acc: {best_val_acc:.1%}  →  stance_classifier_best.pt")
print(f"Final weights →  stance_classifier.pt")

# ── Confusion matrix (uses best checkpoint) ────────────────
model.load_state_dict(torch.load('stance_classifier_best.pt'))
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        all_preds.extend(model(X_batch).argmax(1).numpy())
        all_true.extend(y_batch.numpy())

cm = confusion_matrix(all_true, all_preds)
print("\nConfusion matrix (rows=actual, cols=predicted):")
print(f"Classes: {list(le.classes_)}")
print(cm)

print("\nFor feature importance (permutation + SHAP), run analyze_stance_model.py "
      "against the saved stance_classifier_best.pt")

# ── Plots ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Loss curves
axes[0].plot(train_losses, label='train')
axes[0].plot(val_losses,   label='val')
axes[0].set_title('Loss')
axes[0].legend()

# Val accuracy
axes[1].plot(val_accs)
axes[1].set_title('Val Accuracy')
axes[1].set_ylim(0, 1)

# Confusion matrix
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(ax=axes[2], xticks_rotation=45, colorbar=False)
axes[2].set_title('Confusion Matrix (best checkpoint)')

plt.tight_layout()
plt.savefig('stance_training_curves.png')
plt.show()
print("Plots saved to stance_training_curves.png")