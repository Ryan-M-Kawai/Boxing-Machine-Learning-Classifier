import json
import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import shap
import subprocess
import sys
from model import PunchClassifier, StanceClassifier

# ── Choose which model/feature set to analyze ────────────────

user_input = input("Enter '0' to analyze frontal punches, '1' for sideways punches, '2' for stance classification, or '3' for all: ").strip().lower()
if user_input == "0":
    print("Analyzing frontal punches..., quit with Q")
    MODE = "punch"
elif user_input == "1":
    print("Analyzing sideways punches..., quit with Q")
    MODE = "sideways"
elif user_input == "2":
    print("Analyzing stance classification..., quit with Q")
    MODE = "stance"
elif user_input == "3":
    print("Analyzing all three models..., quit with Q")
    for i in range(3):
        subprocess.run([sys.executable, sys.argv[0]], input=f"{i}\n", text=True)
    sys.exit(0)
else:
    raise ValueError("Invalid input. Please enter '0', '1', '2', or '3'.")

# ── Where to save output images ───────────────────────────────
OUTPUT_DIR = os.path.join("analysis_output", MODE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out_path(filename):
    return os.path.join(OUTPUT_DIR, filename)

FEATURE_CONFIGS = {
    "punch": {
        "feature_names": [
            "elbowAngleL", "elbowAngleR", "shoulderAngleL", "shoulderAngleR",
            "wristHeightL", "wristHeightR", "wristLextension_horz", "wristRextension_horz",
            "wristLextension_forward", "wristRextension_forward", "shoulderTilt", "hip_tilt",
            "hipAngleL", "hipAngleR", "legAngleL", "legAngleR",
            "wristAboveElbowL", "wristAboveElbowR", "elbowHeightL", "elbowHeightR",
            "wristLateralL", "wristLateralR", "elbowFlareL", "elbowFlareR",
            "wrist_foot_extensionL", "wrist_foot_extensionR",
        ],
        "data_file": "training_data.json",
        "data_key": "frames",              # punch reps store a (30, features) sequence
        "checkpoint": "punch_classifier_best.pt",
        "encoder_file": "label_encoder.pkl",
        "input_size": 26,
        "model_class": PunchClassifier,
    },
    "sideways": {
        "feature_names": [
            "elbowAngleL", "elbowAngleR", "shoulderAngleL", "shoulderAngleR",
            "wristHeightL", "wristHeightR", "wristLextension_horz", "wristRextension_horz",
            "wristLextension_forward", "wristRextension_forward", "shoulderTilt", "hip_tilt",
            "hipAngleL", "hipAngleR", "legAngleL", "legAngleR",
            "wristAboveElbowL", "wristAboveElbowR", "elbowHeightL", "elbowHeightR",
            "wristLateralL", "wristLateralR", "elbowFlareL", "elbowFlareR",
            "wrist_foot_extensionL", "wrist_foot_extensionR",
        ],
        "data_file": "sideways_training_data.json",
        "data_key": "frames",
        "checkpoint": "punch_classifier_sideways_best.pt",
        "encoder_file": "label_encoder_sideways.pkl",
        "input_size": 26,
        "model_class": PunchClassifier,
    },
    "stance": {
        "feature_names": [
            "foot_x_diff", "foot_z_diff", "shoulder_x_diff", "shoulder_z_diff", "hip_x_diff", "hip_z_diff",
            "direction","foot_x_diff_canon", "shoulder_z_diff_canon", "hip_z_diff_canon",
            ],
        "data_file": "stance_data.json",
        "data_key": "features",            # stance snapshots store a flat feature vector
        "checkpoint": "stance_classifier_best.pt",
        "encoder_file": "stance_label_encoder.pkl",
        "input_size": 10,
        "model_class": StanceClassifier,
    },
}

cfg = FEATURE_CONFIGS[MODE]
FEATURE_NAMES = cfg["feature_names"]
print(f"── Analyzing MODE = '{MODE}' ──")

# ── Load label encoder ──────────────────────────────────────
with open(cfg["encoder_file"], 'rb') as f:
    le = pickle.load(f)
num_classes = len(le.classes_)
print("Classes:", le.classes_)

# ── Load data (same split as training, so val set matches) ──
with open(cfg["data_file"]) as f:
    raw = json.load(f)

X_np = np.array([d[cfg["data_key"]] for d in raw], dtype=np.float32)
labels = [d['label'] for d in raw]
y_np = le.transform(labels)

# is_seq: True for punch data (N, timesteps, features), False for stance (N, features)
is_seq = X_np.ndim == 3
num_features = X_np.shape[-1]

if num_features != len(FEATURE_NAMES):
    print(f"[WARN] {cfg['data_file']} has {num_features} features per sample, "
          f"but FEATURE_NAMES has {len(FEATURE_NAMES)} entries. "
          f"Check that the feature extractor and FEATURE_NAMES are in sync before trusting the rankings below.")

X_train, X_val, y_train, y_val = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
)

# ── Load trained model ───────────────────────────────────────
ModelClass = cfg["model_class"]
model = ModelClass(input_size=num_features, num_classes=num_classes)
model.load_state_dict(torch.load(cfg["checkpoint"]))
model.eval()
print(f"Loaded {cfg['checkpoint']}")

name_col_width = max(len(n) for n in FEATURE_NAMES) + 2

# ── Confusion matrix ─────────────────────────────────────────
all_preds, all_true = [], []
with torch.no_grad():
    for i in range(0, len(X_val), 32):
        xb = torch.tensor(X_val[i:i+32])
        yb = y_val[i:i+32]
        preds = model(xb).argmax(1).numpy()
        all_preds.extend(preds)
        all_true.extend(yb)

cm = confusion_matrix(all_true, all_preds)
print("\nConfusion matrix (rows=actual, cols=predicted):")
print(f"Classes: {list(le.classes_)}")
print(cm)

fig0, ax0 = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(ax=ax0, xticks_rotation=45, colorbar=False)
ax0.set_title(f'Confusion Matrix ({MODE})')

plt.tight_layout()
plt.savefig(out_path('confusion_matrix.png'))
plt.show()
print(f"Saved {out_path('confusion_matrix.png')}")

# 
# ── Feature importance (permutation importance) ─────────────
# Idea: shuffle one feature column across samples (breaking its link to the
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
print(f"\nBaseline val accuracy: {baseline_acc:.1%}")

N_REPEATS = 10
rng = np.random.default_rng(42)
importances = np.zeros(num_features)
importances_std = np.zeros(num_features)

print(f"\nComputing permutation importance ({N_REPEATS} repeats per feature — this may take a moment)...")

for f in range(num_features):
    drops = []
    for _ in range(N_REPEATS):
        X_permuted = X_val.copy()
        perm = rng.permutation(len(X_permuted))
        if is_seq:
            # shuffle this feature channel across samples, all timesteps together
            X_permuted[:, :, f] = X_val[perm, :, f]
        else:
            X_permuted[:, f] = X_val[perm, f]

        X_permuted_t = torch.tensor(X_permuted)
        permuted_acc = compute_val_accuracy(model, X_permuted_t, y_val_t)
        drops.append(baseline_acc - permuted_acc)

    importances[f] = np.mean(drops)
    importances_std[f] = np.std(drops)

order = np.argsort(importances)[::-1]

print("\nPermutation feature importance (accuracy drop when shuffled — higher = more important):")
print(f"{'feature':<{name_col_width}}{'importance':>12}{'  (+/- std)':>12}")
for idx in order:
    name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
    print(f"  {name:<{name_col_width}}{importances[idx]*100:>9.2f}%   (+/- {importances_std[idx]*100:.2f}%)")

fig1, ax = plt.subplots(figsize=(10, 8))
sorted_names = [FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}" for i in order]
sorted_importances = importances[order] * 100
sorted_stds = importances_std[order] * 100

y_pos = np.arange(len(sorted_names))
ax.barh(y_pos, sorted_importances, xerr=sorted_stds, color='steelblue')
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_names)
ax.invert_yaxis()
ax.set_xlabel('Val accuracy drop when feature is shuffled (%)')
ax.set_title(f'Permutation feature importance ({MODE})')
ax.axvline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig(out_path('feature_importance.png'))
plt.show()
print(f"Saved {out_path('feature_importance.png')}")

# ── Feature importance (SHAP, gradient-based) ───────────────
# Idea: unlike permutation importance (which measures accuracy drop from
# breaking a feature), SHAP's GradientExplainer uses the model's actual
# gradients — how much the output logits move as each input feature moves —
# combined with a background reference distribution, to attribute each
# prediction's output back to individual input features (an approximation
# of Integrated Gradients / expected gradients).
print("\nComputing SHAP values (gradient-based attribution)...")

background_size = min(100, len(X_train))
background_idx = rng.choice(len(X_train), size=background_size, replace=False)
background = torch.tensor(X_train[background_idx])

explain_size = min(50, len(X_val))
explain_idx = rng.choice(len(X_val), size=explain_size, replace=False)
X_explain = torch.tensor(X_val[explain_idx])
y_explain = y_val[explain_idx]

explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(X_explain)

# SHAP's return shape has varied across versions: either a list of
# per-class arrays, or a single stacked array with class as the last axis.
if isinstance(shap_values, list):
    shap_arr = np.stack(shap_values, axis=-1)
else:
    shap_arr = np.asarray(shap_values)

if is_seq:
    # shape: (samples, timesteps, features, classes)
    print(f"SHAP values shape (samples, timesteps, features, classes): {shap_arr.shape}")
    shap_importance = np.mean(np.abs(shap_arr), axis=(0, 1, 3))
    per_class_importance = np.zeros((num_classes, num_features))
    for c in range(num_classes):
        mask = y_explain == c
        if mask.sum() > 0:
            per_class_importance[c] = np.mean(np.abs(shap_arr[mask, :, :, c]), axis=(0, 1))
else:
    # shape: (samples, features, classes)
    print(f"SHAP values shape (samples, features, classes): {shap_arr.shape}")
    shap_importance = np.mean(np.abs(shap_arr), axis=(0, 2))
    per_class_importance = np.zeros((num_classes, num_features))
    for c in range(num_classes):
        mask = y_explain == c
        if mask.sum() > 0:
            per_class_importance[c] = np.mean(np.abs(shap_arr[mask, :, c]), axis=0)

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
ax.set_title(f'SHAP feature importance ({MODE}, overall)')

plt.tight_layout()
plt.savefig(out_path('shap_feature_importance.png'))
plt.show()
print(f"Saved {out_path('shap_feature_importance.png')}")

# ── Plot: per-class SHAP importance heatmap ─────────────────
fig4, ax = plt.subplots(figsize=(10, max(6, num_classes * 0.6)))
im = ax.imshow(per_class_importance, aspect='auto', cmap='viridis')
ax.set_xticks(np.arange(num_features))
ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=8)
ax.set_yticks(np.arange(num_classes))
ax.set_yticklabels(le.classes_)
ax.set_title(f'SHAP importance per {MODE} class')
fig4.colorbar(im, ax=ax, label='mean |SHAP value|')

plt.tight_layout()
plt.savefig(out_path('shap_per_class_importance.png'))
plt.show()
print(f"Saved {out_path('shap_per_class_importance.png')}")

# ── Feature correlation matrix ───────────────────────────────
if is_seq:
    # Flatten (samples, timesteps, features) -> (samples*timesteps, features)
    # so each row is one frame's feature vector, treated as one observation.
    X_flat = X_np.reshape(-1, X_np.shape[-1])
else:
    X_flat = X_np

corr_matrix = np.corrcoef(X_flat, rowvar=False)  # shape (features, features)

fig5, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(FEATURE_NAMES)))
ax.set_yticks(np.arange(len(FEATURE_NAMES)))
ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=8)
ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
ax.set_title(f'Feature correlation matrix ({MODE})')
fig5.colorbar(im, ax=ax, label='Pearson correlation')

for i in range(len(FEATURE_NAMES)):
    for j in range(len(FEATURE_NAMES)):
        val = corr_matrix[i, j]
        if abs(val) > 0.3 and i != j:
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=6, color='black')

plt.tight_layout()
plt.savefig(out_path('feature_correlation.png'))
plt.show()
print(f"Saved {out_path('feature_correlation.png')}")