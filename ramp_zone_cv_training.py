from __future__ import annotations

import json
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _parse_dataset_zone_part(dataset_csv_name: str) -> tuple[str, str]:
    match = re.match(r"^cnn_dataset_zone(?P<zone>\d{2})_(?P<part>p\d+|full)\.csv$", dataset_csv_name)
    if not match:
        raise ValueError(f"Cannot parse zone/part from dataset file name: {dataset_csv_name}")
    zone = match.group("zone")
    part_token = match.group("part")
    part = "full" if part_token == "full" else f"part{int(part_token[1:])}"
    return zone, part


def _load_dataset_frames(file_list: list[Path]) -> pd.DataFrame:
    frames = []
    seen_names = {}
    for csv_path in file_list:
        if csv_path.name in seen_names:
            raise ValueError(
                f"Duplicate dataset file name found for CV: {csv_path.name} "
                f"in {seen_names[csv_path.name]} and {csv_path}"
            )
        seen_names[csv_path.name] = csv_path
        zone, part = _parse_dataset_zone_part(csv_path.name)
        frame = pd.read_csv(csv_path).copy()
        frame = frame.assign(
            dataset_csv_path=str(csv_path.resolve()),
            dataset_csv_name=csv_path.name,
            cv_zone=zone,
            cv_part=part,
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _apply_label_caps(frame: pd.DataFrame, cap_map: dict, seed: int) -> tuple[pd.DataFrame, list[tuple]]:
    cap_records = []
    capped_frames = []
    for label_name, group in frame.groupby("label", sort=False):
        cap = cap_map.get(label_name, None)
        original_n = len(group)
        if cap is None or cap >= original_n:
            kept_group = group.copy()
            kept_n = original_n
        else:
            kept_group = group.sample(n=int(cap), random_state=seed).copy()
            kept_n = len(kept_group)
        capped_frames.append(kept_group)
        cap_records.append((label_name, original_n, kept_n, cap))
    out = pd.concat(capped_frames, ignore_index=True) if capped_frames else frame.copy()
    return out, cap_records


def _impute_scalar_features(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame | None,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, float]]:
    train_frame = train_frame.copy()
    val_frame = None if val_frame is None else val_frame.copy()
    impute_values = {}
    for col in feature_cols:
        train_frame[col] = pd.to_numeric(train_frame[col], errors="coerce")
        train_values = train_frame[col].to_numpy(dtype=np.float32)
        train_finite = np.isfinite(train_values)
        value = float(train_values[train_finite].mean()) if train_finite.any() else 0.0
        impute_values[col] = value
        train_frame.loc[~train_finite, col] = value
        if val_frame is not None:
            val_frame[col] = pd.to_numeric(val_frame[col], errors="coerce")
            val_values = val_frame[col].to_numpy(dtype=np.float32)
            val_finite = np.isfinite(val_values)
            val_frame.loc[~val_finite, col] = value
    return train_frame, val_frame, impute_values


def _augment_ramp_rows(train_frame: pd.DataFrame, z_cols: list[str], cfg: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = train_frame.copy().reset_index(drop=True)
    train_frame["is_augmented"] = 0
    train_frame["aug_parent_id"] = np.arange(len(train_frame))
    if (not cfg["ENABLE_RAMP_AUG"]) or cfg["RAMP_AUG_FACTOR"] <= 0:
        return train_frame, pd.DataFrame()

    rng = np.random.default_rng(seed)
    ramp_df = train_frame[train_frame["label"] == "RAMP"].copy()
    aug_rows = []
    for _, row in ramp_df.iterrows():
        parent_id = int(row["aug_parent_id"])
        last_valid_bin = int(row["last_valid_bin"])
        valid_len = last_valid_bin + 1
        if valid_len <= 0:
            continue
        base_z = row[z_cols].to_numpy(dtype=float)
        for _ in range(cfg["RAMP_AUG_FACTOR"]):
            z_aug = base_z.copy()
            offset = rng.uniform(-cfg["RAMP_OFFSET_RANGE"], cfg["RAMP_OFFSET_RANGE"])
            noise = rng.normal(0.0, cfg["RAMP_NOISE_STD"], size=valid_len)
            z_aug[:valid_len] = z_aug[:valid_len] + offset + noise
            z_aug[valid_len:] = 0.0
            new_row = row.copy()
            for i, col in enumerate(z_cols):
                new_row[col] = float(z_aug[i])
            new_row["is_augmented"] = 1
            new_row["aug_parent_id"] = parent_id
            aug_rows.append(new_row)
    aug_df = pd.DataFrame(aug_rows)
    if aug_df.empty:
        return train_frame, aug_df
    out = pd.concat([train_frame, aug_df], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out, aug_df


def _df_to_arrays(frame: pd.DataFrame, z_cols: list[str], gap_cols: list[str], pad_cols: list[str], feature_cols: list[str]):
    z_arr = frame[z_cols].to_numpy(dtype=np.float32)
    gap_arr = frame[gap_cols].to_numpy(dtype=np.float32)
    pad_arr = frame[pad_cols].to_numpy(dtype=np.float32)
    x_arr = np.stack([z_arr, gap_arr, pad_arr], axis=1)
    x_feat_arr = frame[feature_cols].to_numpy(dtype=np.float32)
    y_arr = frame["target_idx"].to_numpy(dtype=np.int64)
    return x_arr, x_feat_arr, y_arr


class _RampProfileDataset(Dataset):
    def __init__(self, x, x_feat, y, z_mean, z_std, feat_mean, feat_std):
        self.x = x.copy()
        self.x[:, 0, :] = (self.x[:, 0, :] - z_mean) / z_std
        self.x_feat = x_feat.copy()
        self.x_feat = (self.x_feat - feat_mean) / feat_std
        self.y = y.copy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.x_feat[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


class _FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class _SpatialAttention1d(nn.Module):
    def __init__(self, in_ch, reduction=8):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_ch, in_ch // reduction, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(in_ch // reduction, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.attn(x)
        return x * w


class _RampCNN(nn.Module):
    def __init__(self, in_channels=3, num_features=2, num_classes=3, dropout=0.1):
        super().__init__()
        self.use_scalar_features = num_features > 0
        # Kept for checkpoint compatibility with the v1 prediction CLI; the
        # existing v1 forward path defines this module but does not apply it.
        self.attn = _SpatialAttention1d(128, reduction=8)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding="same"),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding="same"),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding="same"),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.profile_head = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.LeakyReLU(0.1))
        if self.use_scalar_features:
            self.feature_head = nn.Sequential(nn.Linear(num_features, 16), nn.LeakyReLU(0.1))
            classifier_in = 64 + 16
        else:
            self.feature_head = None
            classifier_in = 64
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, x_feat):
        x = self.features(x)
        x = self.pool(x)
        x = self.profile_head(x)
        if self.use_scalar_features:
            x_feat = self.feature_head(x_feat)
            x = torch.cat([x, x_feat], dim=1)
        return self.classifier(x)


def _metric_dict(y_true, y_pred, eval_labels, index_to_label):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    present_labels = sorted(np.unique(y_true).tolist())
    out = {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else np.nan,
        "macro_f1_all_classes": float(f1_score(y_true, y_pred, labels=eval_labels, average="macro", zero_division=0)) if len(y_true) else np.nan,
        "macro_f1_present_classes": float(f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0)) if len(y_true) and present_labels else np.nan,
        "present_classes": ",".join(index_to_label[i] for i in present_labels),
    }
    for class_idx, label_name in index_to_label.items():
        true_mask = y_true == class_idx
        pred_mask = y_pred == class_idx
        tp = int((true_mask & pred_mask).sum())
        fp = int((~true_mask & pred_mask).sum())
        fn = int((true_mask & ~pred_mask).sum())
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) else np.nan
        out[f"n_{label_name}"] = int(true_mask.sum())
        out[f"pred_{label_name}"] = int(pred_mask.sum())
        out[f"precision_{label_name}"] = float(precision) if np.isfinite(precision) else np.nan
        out[f"recall_{label_name}"] = float(recall) if np.isfinite(recall) else np.nan
        out[f"f1_{label_name}"] = float(f1) if np.isfinite(f1) else np.nan
    return out


def _run_epoch(model, loader, criterion, optimizer, device, eval_labels, index_to_label):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_count = 0
    all_preds = []
    all_targets = []
    for xb, xb_feat, yb in loader:
        xb = xb.to(device)
        xb_feat = xb_feat.to(device)
        yb = yb.to(device)
        with torch.set_grad_enabled(train_mode):
            logits = model(xb, xb_feat)
            loss = criterion(logits, yb)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * xb.size(0)
        total_count += int(xb.size(0))
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    metrics = _metric_dict(y_true, y_pred, eval_labels, index_to_label)
    metrics["loss"] = total_loss / total_count
    return metrics


def _prepare_objects(train_frame, val_frame, cfg, z_cols, gap_cols, pad_cols, seed):
    train_frame = train_frame.copy().reset_index(drop=True)
    val_frame = None if val_frame is None else val_frame.copy().reset_index(drop=True)
    capped_train, cap_records = _apply_label_caps(train_frame, cfg["MAX_SAMPLES_PER_LABEL"], seed)
    capped_train, val_frame, impute_values = _impute_scalar_features(capped_train, val_frame, cfg["ACTIVE_FEATURE_COLS"])
    train_final, ramp_aug_df = _augment_ramp_rows(capped_train, z_cols, cfg, seed)
    if val_frame is not None:
        val_frame = val_frame.copy().reset_index(drop=True)
        val_frame["is_augmented"] = 0
        val_frame["aug_parent_id"] = -1

    x_train, x_train_feat, y_train = _df_to_arrays(train_final, z_cols, gap_cols, pad_cols, cfg["ACTIVE_FEATURE_COLS"])
    if cfg["USE_Z_SCORE_ON_Z_CHANNEL"]:
        z_train = x_train[:, 0, :]
        z_mean = float(z_train.mean())
        z_std = float(z_train.std())
        if z_std < 1e-8:
            z_std = 1.0
    else:
        z_mean = 0.0
        z_std = 1.0

    if cfg["USE_SCALAR_FEATURES"]:
        feat_mean = x_train_feat.mean(axis=0).astype(np.float32)
        feat_std = x_train_feat.std(axis=0).astype(np.float32)
        feat_std[feat_std < 1e-8] = 1.0
    else:
        feat_mean = np.zeros(0, dtype=np.float32)
        feat_std = np.ones(0, dtype=np.float32)

    class_counts = np.bincount(y_train, minlength=len(cfg["VALID_CLASS_IDS"]))
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=cfg["device"])

    train_ds = _RampProfileDataset(x_train, x_train_feat, y_train, z_mean, z_std, feat_mean, feat_std)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=cfg["BATCH_SIZE"], shuffle=True, generator=loader_generator)

    val_loader = None
    if val_frame is not None:
        x_val, x_val_feat, y_val = _df_to_arrays(val_frame, z_cols, gap_cols, pad_cols, cfg["ACTIVE_FEATURE_COLS"])
        val_ds = _RampProfileDataset(x_val, x_val_feat, y_val, z_mean, z_std, feat_mean, feat_std)
        val_loader = DataLoader(val_ds, batch_size=cfg["BATCH_SIZE"], shuffle=False)

    return {
        "train_frame": train_final,
        "val_frame": val_frame,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "z_mean": z_mean,
        "z_std": z_std,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "class_weights_t": class_weights_t,
        "class_weights": class_weights,
        "class_counts": class_counts,
        "feature_impute_values": impute_values,
        "cap_records": cap_records,
        "ramp_aug_rows": len(ramp_aug_df),
    }


def _train_one_model(train_frame, val_frame, fixed_epochs, fold_name, cfg, z_cols, gap_cols, pad_cols, seed):
    _seed_everything(seed)
    prepared = _prepare_objects(train_frame, val_frame, cfg, z_cols, gap_cols, pad_cols, seed)
    model = _RampCNN(
        in_channels=len(cfg["CHANNELS"]),
        num_features=len(cfg["ACTIVE_FEATURE_COLS"]),
        num_classes=len(cfg["VALID_CLASS_IDS"]),
        dropout=cfg["DROPOUT"],
    ).to(cfg["device"])
    if cfg["USE_FOCAL_LOSS"]:
        criterion = _FocalLoss(gamma=cfg["FOCAL_GAMMA"], weight=prepared["class_weights_t"])
    else:
        criterion = nn.CrossEntropyLoss(weight=prepared["class_weights_t"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["LEARNING_RATE"], weight_decay=cfg["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    history_rows = []
    best_state = None
    best_epoch = -1
    best_val_loss = float("inf")
    best_metric = -float("inf")
    epochs_without_improve = 0
    max_epochs = int(fixed_epochs) if fixed_epochs is not None else int(cfg["EPOCHS"])

    for epoch in range(1, max_epochs + 1):
        train_metrics = _run_epoch(model, prepared["train_loader"], criterion, optimizer, cfg["device"], cfg["EVAL_LABELS"], cfg["INDEX_TO_LABEL"])
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_f1_all_classes": train_metrics["macro_f1_all_classes"],
            "train_f1_present_classes": train_metrics["macro_f1_present_classes"],
        }
        if prepared["val_loader"] is not None:
            val_metrics = _run_epoch(model, prepared["val_loader"], criterion, None, cfg["device"], cfg["EVAL_LABELS"], cfg["INDEX_TO_LABEL"])
            scheduler.step(val_metrics["loss"])
            row.update({
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_f1_all_classes": val_metrics["macro_f1_all_classes"],
                "val_f1_present_classes": val_metrics["macro_f1_present_classes"],
            })
            selection_metric_name = cfg["CV_SELECTION_METRIC"]
            if selection_metric_name not in val_metrics:
                raise ValueError(f"Unsupported CV_SELECTION_METRIC={selection_metric_name!r}")
            selection_metric = val_metrics[selection_metric_name]
            if selection_metric > best_metric:
                best_metric = selection_metric
                best_val_loss = val_metrics["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= cfg["EARLY_STOPPING_PATIENCE"]:
                    print(f"{fold_name}: early stopping at epoch {epoch}; best_epoch={best_epoch}")
                    history_rows.append(row)
                    break
        else:
            row.update({"val_loss": np.nan, "val_acc": np.nan, "val_f1_all_classes": np.nan, "val_f1_present_classes": np.nan})
            best_epoch = epoch
            best_metric = np.nan
        history_rows.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, prepared, pd.DataFrame(history_rows), {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss) if np.isfinite(best_val_loss) else np.nan,
        "best_val_present_f1": float(best_metric) if np.isfinite(best_metric) else np.nan,
    }


def _predict_frame(model, frame, prepared, cfg, z_cols, gap_cols, pad_cols, split_name, holdout_zone=None):
    frame = frame.copy().reset_index(drop=True)
    frame["is_augmented"] = 0
    frame["aug_parent_id"] = -1
    x_eval, x_feat_eval, y_eval = _df_to_arrays(frame, z_cols, gap_cols, pad_cols, cfg["ACTIVE_FEATURE_COLS"])
    eval_ds = _RampProfileDataset(x_eval, x_feat_eval, y_eval, prepared["z_mean"], prepared["z_std"], prepared["feat_mean"], prepared["feat_std"])
    eval_loader = DataLoader(eval_ds, batch_size=cfg["BATCH_SIZE"], shuffle=False)
    model.eval()
    logits_parts = []
    with torch.no_grad():
        for xb, xb_feat, _ in eval_loader:
            xb = xb.to(cfg["device"])
            xb_feat = xb_feat.to(cfg["device"])
            logits_parts.append(model(xb, xb_feat).cpu().numpy())
    logits = np.concatenate(logits_parts, axis=0)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    pred_idx = probs.argmax(axis=1)
    out = frame.copy().reset_index(drop=True)
    out["split"] = split_name
    out["holdout_zone"] = holdout_zone
    out["true_idx"] = y_eval
    out["pred_idx"] = pred_idx
    out["true_label"] = out["true_idx"].map(cfg["INDEX_TO_LABEL"])
    out["pred_label"] = out["pred_idx"].map(cfg["INDEX_TO_LABEL"])
    out["pred_confidence"] = probs.max(axis=1)
    out["correct"] = out["pred_idx"] == out["true_idx"]
    for class_idx, label_name in cfg["INDEX_TO_LABEL"].items():
        out[f"prob_{label_name}"] = probs[:, class_idx]
    return out


def _save_confusion_matrices(pred_df: pd.DataFrame, output_dir: Path, cfg: dict) -> tuple[Path, Path]:
    y_true = pred_df["true_idx"].to_numpy(dtype=int)
    y_pred = pred_df["pred_idx"].to_numpy(dtype=int)
    display_names = [cfg["INDEX_TO_LABEL"][i].replace("_", "\n") for i in cfg["EVAL_LABELS"]]
    cm = confusion_matrix(y_true, y_pred, labels=cfg["EVAL_LABELS"])
    cm_norm = confusion_matrix(y_true, y_pred, labels=cfg["EVAL_LABELS"], normalize="true")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=display_names, yticklabels=display_names, ax=ax)
    ax.set_title("Zone CV Held-Out Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = output_dir / "zone_cv_confusion_matrix.png"
    fig.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.show()

    fig_norm, ax_norm = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=display_names, yticklabels=display_names, ax=ax_norm)
    ax_norm.set_title("Zone CV Held-Out Confusion Matrix (Normalized)")
    ax_norm.set_xlabel("Predicted")
    ax_norm.set_ylabel("True")
    plt.tight_layout()
    cm_norm_path = output_dir / "zone_cv_confusion_matrix_normalized.png"
    fig_norm.savefig(cm_norm_path, dpi=200, bbox_inches="tight")
    plt.show()
    return cm_path, cm_norm_path


def run_zone_holdout_cv(ns: dict) -> dict:
    cfg = {name: ns[name] for name in [
        "DATASET_DIR", "TRAIN_DIR", "VAL_DIR", "DATASET_GLOB", "OUTPUT_DIR", "N_BINS", "MODEL_VERSION",
        "CHANNELS", "FEATURE_COLS", "USE_SCALAR_FEATURES", "ACTIVE_FEATURE_COLS", "LABEL_MAP", "VALID_CLASS_IDS",
        "CLASS_TO_INDEX", "INDEX_TO_CLASS", "INDEX_TO_LABEL", "BATCH_SIZE", "LEARNING_RATE", "WEIGHT_DECAY",
        "EPOCHS", "EARLY_STOPPING_PATIENCE", "DROPOUT", "RANDOM_SEED", "USE_Z_SCORE_ON_Z_CHANNEL",
        "USE_FOCAL_LOSS", "FOCAL_GAMMA", "ENABLE_RAMP_AUG", "RAMP_AUG_FACTOR", "RAMP_NOISE_STD",
        "RAMP_OFFSET_RANGE", "MAX_SAMPLES_PER_LABEL", "SPLIT_MODE", "CV_HOLDOUT_ZONES", "CV_SELECTION_METRIC",
        "TRAIN_FINAL_MODEL_AFTER_CV", "FINAL_EPOCH_REDUCTION", "device",
    ]}
    cfg["EVAL_LABELS"] = list(range(len(cfg["INDEX_TO_LABEL"])))
    cfg["TARGET_LABELS"] = [cfg["INDEX_TO_LABEL"][i] for i in cfg["EVAL_LABELS"]]

    output_dir = Path(cfg["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    z_cols = [f"z_{i:03d}" for i in range(cfg["N_BINS"])]
    gap_cols = [f"gap_{i:03d}" for i in range(cfg["N_BINS"])]
    pad_cols = [f"pad_{i:03d}" for i in range(cfg["N_BINS"])]

    all_files = sorted(list(Path(cfg["TRAIN_DIR"]).glob(cfg["DATASET_GLOB"])) + list(Path(cfg["VAL_DIR"]).glob(cfg["DATASET_GLOB"])))
    if not all_files:
        raise FileNotFoundError(f"No dataset CSVs found in {cfg['TRAIN_DIR']} or {cfg['VAL_DIR']} matching {cfg['DATASET_GLOB']!r}")
    df_raw = _load_dataset_frames(all_files)
    required_cols = z_cols + gap_cols + pad_cols + cfg["ACTIVE_FEATURE_COLS"] + ["class_int", "label", "side", "s_m", "source_file", "dataset_csv_path", "cv_zone"]
    missing = [col for col in required_cols if col not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns for zone CV: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    df = df_raw[df_raw["class_int"].isin(cfg["VALID_CLASS_IDS"])].copy()
    df["target_idx"] = df["class_int"].map(cfg["CLASS_TO_INDEX"]).astype(int)
    discovered_zones = sorted(df["cv_zone"].dropna().unique().tolist())
    holdout_zones = discovered_zones if cfg["CV_HOLDOUT_ZONES"] is None else [str(zone).zfill(2) for zone in cfg["CV_HOLDOUT_ZONES"]]
    unknown_zones = sorted(set(holdout_zones) - set(discovered_zones))
    if unknown_zones:
        raise ValueError(f"CV_HOLDOUT_ZONES contains zones not found in datasets: {unknown_zones}")

    print(f"Zone-wise CV output dir: {output_dir}")
    print(f"Dataset files loaded: {len(all_files):,}")
    print(f"Rows loaded: {len(df):,}")
    print(f"Discovered zones: {discovered_zones}")
    print(f"Holdout zones: {holdout_zones}")
    print("Rows by zone:")
    print(df.groupby(["cv_zone", "label"]).size().unstack(fill_value=0).to_string())

    fold_summaries = []
    fold_predictions = []
    best_epochs = []
    for fold_idx, holdout_zone in enumerate(holdout_zones, start=1):
        print(f"\n--- Zone CV fold {fold_idx}/{len(holdout_zones)} | holdout zone {holdout_zone} ---")
        train_fold = df[df["cv_zone"] != holdout_zone].copy()
        val_fold = df[df["cv_zone"] == holdout_zone].copy()
        if train_fold.empty or val_fold.empty:
            raise ValueError(f"Invalid fold for zone {holdout_zone}: train={len(train_fold)} val={len(val_fold)}")
        model, prepared, history, fit_info = _train_one_model(train_fold, val_fold, None, f"zone{holdout_zone}", cfg, z_cols, gap_cols, pad_cols, cfg["RANDOM_SEED"])
        val_pred = _predict_frame(model, prepared["val_frame"], prepared, cfg, z_cols, gap_cols, pad_cols, "val", holdout_zone)
        y_true = val_pred["true_idx"].to_numpy(dtype=int)
        y_pred = val_pred["pred_idx"].to_numpy(dtype=int)
        metrics = _metric_dict(y_true, y_pred, cfg["EVAL_LABELS"], cfg["INDEX_TO_LABEL"])
        best_epochs.append(fit_info["best_epoch"])
        fold_predictions.append(val_pred)
        row = {
            "row_type": "fold",
            "holdout_zone": holdout_zone,
            "n_train_files": int(train_fold["dataset_csv_name"].nunique()),
            "n_val_files": int(val_fold["dataset_csv_name"].nunique()),
            "train_n": int(len(train_fold)),
            "val_n": int(len(val_fold)),
            "best_epoch": fit_info["best_epoch"],
            "best_val_loss": fit_info["best_val_loss"],
            **metrics,
        }
        fold_summaries.append(row)
        print(
            f"holdout_zone={holdout_zone} train_n={len(train_fold):,} val_n={len(val_fold):,} "
            f"best_epoch={fit_info['best_epoch']} val_acc={metrics['accuracy']:.4f} "
            f"val_present_f1={metrics['macro_f1_present_classes']:.4f} present={metrics['present_classes']}"
        )

    heldout_pred = pd.concat(fold_predictions, ignore_index=True)
    y_true_cv = heldout_pred["true_idx"].to_numpy(dtype=int)
    y_pred_cv = heldout_pred["pred_idx"].to_numpy(dtype=int)
    aggregate_metrics = _metric_dict(y_true_cv, y_pred_cv, cfg["EVAL_LABELS"], cfg["INDEX_TO_LABEL"])
    if cfg["FINAL_EPOCH_REDUCTION"] != "median":
        raise ValueError(f"Unsupported FINAL_EPOCH_REDUCTION={cfg['FINAL_EPOCH_REDUCTION']!r}; only 'median' is implemented.")
    final_epochs = max(1, int(round(float(np.median(best_epochs)))))

    print("\n=== Zone-wise CV aggregate held-out metrics ===")
    print(f"Held-out rows: {len(heldout_pred):,}")
    print(f"Accuracy: {aggregate_metrics['accuracy']:.4f}")
    print(f"Macro F1 present classes: {aggregate_metrics['macro_f1_present_classes']:.4f}")
    print(f"Macro F1 all classes: {aggregate_metrics['macro_f1_all_classes']:.4f}")
    print()
    print(classification_report(y_true_cv, y_pred_cv, labels=cfg["EVAL_LABELS"], target_names=cfg["TARGET_LABELS"], digits=4, zero_division=0))
    summary_df = pd.DataFrame(fold_summaries)
    print("Fold summary:")
    print(summary_df[["holdout_zone", "train_n", "val_n", "best_epoch", "accuracy", "macro_f1_present_classes", "macro_f1_all_classes", "present_classes"]].to_string(index=False))
    print(f"Best epochs by fold: {best_epochs}")
    print(f"Median best epoch for final training: {final_epochs}")

    summary_path = output_dir / "zone_cv_summary.csv"
    aggregate_path = output_dir / "zone_cv_aggregate_summary.csv"
    pred_path = output_dir / "zone_cv_heldout_predictions_by_row.csv"
    summary_df.to_csv(summary_path, index=False)
    aggregate_row = {
        "row_type": "overall_heldout_aggregate",
        "n_rows": int(len(heldout_pred)),
        "fold_count": int(len(holdout_zones)),
        "holdout_zones": ",".join(holdout_zones),
        "cv_best_epochs": ",".join(str(int(epoch)) for epoch in best_epochs),
        "cv_median_best_epoch": int(final_epochs),
        **aggregate_metrics,
    }
    pd.DataFrame([aggregate_row]).to_csv(aggregate_path, index=False)
    row_output_cols = [
        "holdout_zone", "dataset_csv_name", "source_file", "s_m", "side",
        "true_label", "pred_label", "pred_confidence", "correct",
        "class_int", "target_idx", "pred_idx",
    ] + [f"prob_{label_name}" for label_name in cfg["INDEX_TO_LABEL"].values()]
    row_output_cols = [col for col in row_output_cols if col in heldout_pred.columns]
    heldout_pred[row_output_cols].to_csv(pred_path, index=False)
    cm_path, cm_norm_path = _save_confusion_matrices(heldout_pred, output_dir, cfg)

    result = {
        "cv_summary": summary_df,
        "cv_aggregate_metrics": aggregate_metrics,
        "cv_best_epochs": [int(epoch) for epoch in best_epochs],
        "final_epochs": int(final_epochs),
        "heldout_predictions": heldout_pred,
        "summary_path": summary_path,
        "aggregate_path": aggregate_path,
        "predictions_path": pred_path,
        "confusion_matrix_path": cm_path,
        "confusion_matrix_normalized_path": cm_norm_path,
    }

    if cfg["TRAIN_FINAL_MODEL_AFTER_CV"]:
        print(f"\n=== Training final model on all zones for {final_epochs} epochs ===")
        final_model, final_prepared, final_history, _ = _train_one_model(df, None, final_epochs, "final_all_zones", cfg, z_cols, gap_cols, pad_cols, cfg["RANDOM_SEED"])
        model_path = output_dir / "ramp_cnn_pytorch.pt"
        meta_path = output_dir / "ramp_cnn_pytorch_meta.json"
        final_history_path = output_dir / "final_training_history.csv"
        final_curves_path = output_dir / "final_training_curves.png"
        final_history.to_csv(final_history_path, index=False)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(final_history["epoch"], final_history["train_loss"])
        axes[0].set_title("Final Train Loss")
        axes[0].grid(True)
        axes[1].plot(final_history["epoch"], final_history["train_acc"])
        axes[1].set_title("Final Train Accuracy")
        axes[1].grid(True)
        axes[2].plot(final_history["epoch"], final_history["train_f1_present_classes"])
        axes[2].set_title("Final Train Present-Class F1")
        axes[2].grid(True)
        plt.tight_layout()
        fig.savefig(final_curves_path, dpi=200, bbox_inches="tight")
        plt.show()

        torch.save({
            "model_state_dict": final_model.state_dict(),
            "class_to_index": cfg["CLASS_TO_INDEX"],
            "index_to_label": cfg["INDEX_TO_LABEL"],
            "channels": cfg["CHANNELS"],
            "use_scalar_features": cfg["USE_SCALAR_FEATURES"],
            "feature_cols": cfg["ACTIVE_FEATURE_COLS"],
            "available_feature_cols": cfg["FEATURE_COLS"],
            "n_bins": cfg["N_BINS"],
            "model_version": cfg["MODEL_VERSION"],
        }, model_path)
        meta = {
            "dataset_dir": str(cfg["DATASET_DIR"]),
            "train_dir": str(cfg["TRAIN_DIR"]),
            "val_dir": str(cfg["VAL_DIR"]),
            "dataset_glob": cfg["DATASET_GLOB"],
            "dataset_files": [str(path.resolve()) for path in all_files],
            "n_dataset_files": len(all_files),
            "output_dir": str(output_dir),
            "model_version": cfg["MODEL_VERSION"],
            "split_mode": cfg["SPLIT_MODE"],
            "selection_metric": cfg["CV_SELECTION_METRIC"],
            "channels": cfg["CHANNELS"],
            "use_scalar_features": cfg["USE_SCALAR_FEATURES"],
            "feature_cols": cfg["ACTIVE_FEATURE_COLS"],
            "available_feature_cols": cfg["FEATURE_COLS"],
            "n_bins": cfg["N_BINS"],
            "class_to_index": cfg["CLASS_TO_INDEX"],
            "index_to_label": cfg["INDEX_TO_LABEL"],
            "z_mean": final_prepared["z_mean"],
            "z_std": final_prepared["z_std"],
            "feat_mean": final_prepared["feat_mean"].tolist(),
            "feat_std": final_prepared["feat_std"].tolist(),
            "batch_size": cfg["BATCH_SIZE"],
            "learning_rate": cfg["LEARNING_RATE"],
            "weight_decay": cfg["WEIGHT_DECAY"],
            "epochs_requested": cfg["EPOCHS"],
            "early_stopping_patience": cfg["EARLY_STOPPING_PATIENCE"],
            "cv_holdout_zones": holdout_zones,
            "cv_best_epochs": [int(epoch) for epoch in best_epochs],
            "cv_median_best_epoch": int(final_epochs),
            "final_train_epochs": int(final_epochs),
            "final_trained_on_all_zones": True,
            "cv_aggregate_accuracy": aggregate_metrics["accuracy"],
            "cv_aggregate_macro_f1_present_classes": aggregate_metrics["macro_f1_present_classes"],
            "cv_aggregate_macro_f1_all_classes": aggregate_metrics["macro_f1_all_classes"],
            "cv_summary_csv": str(summary_path),
            "cv_aggregate_summary_csv": str(aggregate_path),
            "cv_heldout_predictions_csv": str(pred_path),
            "final_training_history_csv": str(final_history_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved final CV-selected model -> {model_path}")
        print(f"Saved final model metadata    -> {meta_path}")
        result.update({
            "model_path": model_path,
            "meta_path": meta_path,
            "final_training_history_path": final_history_path,
            "final_training_curves_path": final_curves_path,
        })

    print(f"Saved CV fold summary         -> {summary_path}")
    print(f"Saved CV aggregate summary    -> {aggregate_path}")
    print(f"Saved CV held-out predictions -> {pred_path}")
    print(f"Saved CV confusion matrix     -> {cm_path}")
    print(f"Saved CV normalized matrix    -> {cm_norm_path}")
    return result
