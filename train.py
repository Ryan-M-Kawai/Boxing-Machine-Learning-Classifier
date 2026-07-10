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
import shap

# ── Feature names (must match extract_features() order in get_values.py) ──
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

# ── Load data ──────────────────────────────────────────────
with open('training_data.json') as f:
    raw = json.load(f)

X_np = np.array([d['frames'] for d in raw], dtype=np.float32)  # (N, 30, features)
labels = [d['label'] for d in raw]

num_features_in_data = X_np.shape[2]
if num_features_in_data != len(FEATURE_NAMES):
    print(f"[WARN] training_data.json has {num_features_in_data} features per frame, "
          f"but FEATURE_NAMES has {len(FEATURE_NAMES)} entries. "
          f"Check that extract_features() and FEATURE_NAMES are in sync before trusting the importance ranking below.")

le = LabelEncoder()
y_np = le.fit_transform(labels)
print("Classes:", le.classes_)
print("Samples per class:", np.bincount(y_np))

with open('label_encoder.pkl', 'wb') as f:
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
class PunchClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=7):
        super().__init__()
        self.lstm  = nn.LSTM(input_size, hidden_size, batch_first=True)  # dropout moved out
        self.drop1 = nn.Dropout(0.3)   # after LSTM
        self.fc1   = nn.Linear(hidden_size, 32)
        self.relu  = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)   # after fc1
        self.fc2   = nn.Linear(32, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc2(self.drop2(self.relu(self.fc1(self.drop1(out[:, -1, :])))))
        return out

num_features = X_np.shape[2]
num_classes  = len(le.classes_)
model = PunchClassifier(num_features, num_classes=num_classes)

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
        torch.save(model.state_dict(), 'punch_classifier_best.pt')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} — best val acc: {best_val_acc:.1%}")
            break

# Save final weights too
torch.save(model.state_dict(), 'punch_classifier.pt')
print(f"\nBest val acc: {best_val_acc:.1%}  →  punch_classifier_best.pt")
print(f"Final weights →  punch_classifier.pt")

# ── Confusion matrix (uses best checkpoint) ────────────────
model.load_state_dict(torch.load('punch_classifier_best.pt'))
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

# ── Feature importance (permutation importance) ────────────
# Idea: shuffle one feature channel across samples (breaking its link to the
# label while preserving its marginal distribution and every other feature's
# structure), then measure how much validation accuracy drops. A feature the
# model actually relies on will hurt accuracy a lot when permuted; an unused
# feature will barely move the needle. Repeated several times and averaged
# for stability since the shuffle is random.
def compute_val_accuracy(model, X_tensor, y_tensor, batch_size=32):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            xb = X_tensor[i:i+batch_size]
            yb = y_tensor[i:i+batch_size]
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
    return correct / len(X_tensor)

X_val_t = torch.tensor(X_val)
y_val_t = torch.tensor(y_val, dtype=torch.long)

baseline_acc = compute_val_accuracy(model, X_val_t, y_val_t)
print(f"\nBaseline val accuracy (best checkpoint): {baseline_acc:.1%}")

N_REPEATS = 10
rng = np.random.default_rng(42)
importances = np.zeros(num_features)
importances_std = np.zeros(num_features)

print("\nComputing permutation importance "
      f"({N_REPEATS} repeats per feature — this may take a moment)...")

for f in range(num_features):
    drops = []
    for _ in range(N_REPEATS):
        X_permuted = X_val.copy()
        perm = rng.permutation(len(X_permuted))
        # shuffle this feature channel across samples, all 30 timesteps together,
        # so we're permuting "which rep this feature's time series came from"
        X_permuted[:, :, f] = X_val[perm, :, f]

        X_permuted_t = torch.tensor(X_permuted)
        permuted_acc = compute_val_accuracy(model, X_permuted_t, y_val_t)
        drops.append(baseline_acc - permuted_acc)

    importances[f] = np.mean(drops)
    importances_std[f] = np.std(drops)

# ── Print ranked importance ─────────────────────────────────
order = np.argsort(importances)[::-1]
name_col_width = max(len(n) for n in FEATURE_NAMES) + 2

print("\nFeature importance (accuracy drop when shuffled — higher = more important):")
print(f"{'feature':<{name_col_width}}{'importance':>12}{'  (+/- std)':>12}")
for idx in order:
    name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
    print(f"  {name:<{name_col_width}}{importances[idx]*100:>9.2f}%   (+/- {importances_std[idx]*100:.2f}%)")

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
plt.savefig('training_curves.png')
plt.show()
print("Plots saved to training_curves.png")

# ── Separate figure for feature importance ──────────────────
fig2, ax = plt.subplots(figsize=(10, 8))
sorted_names = [FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}" for i in order]
sorted_importances = importances[order] * 100
sorted_stds = importances_std[order] * 100

y_pos = np.arange(len(sorted_names))
ax.barh(y_pos, sorted_importances, xerr=sorted_stds, color='steelblue')
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_names)
ax.invert_yaxis()  # highest importance at top
ax.set_xlabel('Val accuracy drop when feature is shuffled (%)')
ax.set_title('Permutation feature importance')
ax.axvline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("Feature importance plot saved to feature_importance.png")

# ── Feature importance (SHAP, gradient-based) ───────────────
# Idea: unlike permutation importance (which measures accuracy drop from
# breaking a feature), SHAP's GradientExplainer uses the model's actual
# gradients — how much the output logits move as each input feature moves —
# combined with a background reference distribution, to attribute each
# prediction's output back to individual input features (an approximation
# of Integrated Gradients / expected gradients). This tells you not just
# "does the model use this feature" but "how much does this feature push
# the output score, on average, across real predictions."
print("\nComputing SHAP values (gradient-based attribution)...")

# Background: a reference sample of training data GradientExplainer
# integrates gradients against. 100 samples is enough to approximate the
# expectation without being too slow.
background_size = min(100, len(X_train))
background_idx = rng.choice(len(X_train), size=background_size, replace=False)
background = torch.tensor(X_train[background_idx])

# Samples to explain: a subset of validation data (gradient explainer scales
# with sample count, so keep this modest for a live-coach-sized model).
explain_size = min(50, len(X_val))
explain_idx = rng.choice(len(X_val), size=explain_size, replace=False)
X_explain = torch.tensor(X_val[explain_idx])
y_explain = y_val[explain_idx]

model.eval()
explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(X_explain)

# SHAP's return shape has varied across versions: either a list of
# per-class arrays, or a single stacked array with class as the last axis.
# Normalize to shape (samples, timesteps, features, classes).
if isinstance(shap_values, list):
    shap_arr = np.stack(shap_values, axis=-1)
else:
    shap_arr = np.asarray(shap_values)

print(f"SHAP values shape (samples, timesteps, features, classes): {shap_arr.shape}")

# Overall importance: mean absolute SHAP value per feature, averaged over
# samples, timesteps, and classes. Uses magnitude only — direction (does a
# feature push toward or away from a class) is available per-class below
# if you want to dig into a specific punch.
shap_importance = np.mean(np.abs(shap_arr), axis=(0, 1, 3))

# Per-predicted-class importance: for each sample, only look at the SHAP
# values for that sample's true class, so "what mattered for uppercuts"
# isn't diluted by classes that weren't being predicted.
per_class_importance = np.zeros((num_classes, num_features))
for c in range(num_classes):
    mask = y_explain == c
    if mask.sum() > 0:
        per_class_importance[c] = np.mean(np.abs(shap_arr[mask, :, :, c]), axis=(0, 1))

# ── Print ranked SHAP importance ────────────────────────────
shap_order = np.argsort(shap_importance)[::-1]
print("\nSHAP feature importance (mean |gradient-based attribution| — higher = more important):")
print(f"{'feature':<{name_col_width}}{'mean |SHAP|':>14}")
for idx in shap_order:
    name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
    print(f"  {name:<{name_col_width}}{shap_importance[idx]:>12.4f}")

# ── Compare SHAP vs permutation rankings ────────────────────
print("\nSHAP rank vs permutation rank (feature: shap_rank, perm_rank):")
perm_rank = {idx: rank for rank, idx in enumerate(order, start=1)}
shap_rank = {idx: rank for rank, idx in enumerate(shap_order, start=1)}
for idx in shap_order:
    name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
    print(f"  {name:<{name_col_width}} shap #{shap_rank[idx]:<4} perm #{perm_rank[idx]}")

# ── Plot: overall SHAP importance ───────────────────────────
fig3, ax = plt.subplots(figsize=(10, 8))
sorted_shap_names = [FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}" for i in shap_order]
sorted_shap_vals = shap_importance[shap_order]

y_pos = np.arange(len(sorted_shap_names))
ax.barh(y_pos, sorted_shap_vals, color='darkorange')
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_shap_names)
ax.invert_yaxis()
ax.set_xlabel('Mean |SHAP value| (gradient-based attribution)')
ax.set_title('SHAP feature importance (overall)')

plt.tight_layout()
plt.savefig('shap_feature_importance.png')
plt.show()
print("SHAP feature importance plot saved to shap_feature_importance.png")

# ── Plot: per-class SHAP importance heatmap ─────────────────
fig4, ax = plt.subplots(figsize=(10, max(6, num_classes * 0.6)))
im = ax.imshow(per_class_importance, aspect='auto', cmap='viridis')
ax.set_xticks(np.arange(num_features))
ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=8)
ax.set_yticks(np.arange(num_classes))
ax.set_yticklabels(le.classes_)
ax.set_title('SHAP importance per punch class')
fig4.colorbar(im, ax=ax, label='mean |SHAP value|')

plt.tight_layout()
plt.savefig('shap_per_class_importance.png')
plt.show()
print("Per-class SHAP importance heatmap saved to shap_per_class_importance.png")

import numpy as np
import matplotlib.pyplot as plt

# Flatten (samples, timesteps, features) -> (samples*timesteps, features)
# so each row is one frame's feature vector, treated as one observation.
X_flat = X_np.reshape(-1, X_np.shape[2])

corr_matrix = np.corrcoef(X_flat, rowvar=False)  # shape (features, features)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(FEATURE_NAMES)))
ax.set_yticks(np.arange(len(FEATURE_NAMES)))
ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=8)
ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
ax.set_title('Feature correlation matrix')
fig.colorbar(im, ax=ax, label='Pearson correlation')

# Annotate cells with the actual correlation value (skip near-zero for readability)
for i in range(len(FEATURE_NAMES)):
    for j in range(len(FEATURE_NAMES)):
        val = corr_matrix[i, j]
        if abs(val) > 0.3 and i != j:
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=6, color='black')

plt.tight_layout()
plt.savefig('feature_correlation.png')
plt.show()
print("Feature correlation matrix saved to feature_correlation.png")