import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score

def collate(batch):
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    out = {"label": labels}
    if "image" in batch[0]:
        out["image"] = torch.stack([b["image"] for b in batch])
    if "roi_data" in batch[0]:
        out["roi_data"] = torch.stack([b["roi_data"] for b in batch])
    return out

def compute_test_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, num_classes: int):
    """Compute Accuracy (%), AUC (%), Sensitivity (%), Specificity (%), F1-score (%)."""
    from sklearn.metrics import confusion_matrix
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "auc": 0.0, "sensitivity": 0.0, "specificity": 0.0, "f1": 0.0}
    acc = accuracy_score(y_true, y_pred) * 100.0
    # AUC: binary use prob of positive class; multi-class use ovr
    if num_classes == 2:
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            prob_pos = y_prob[:, 1]
        else:
            prob_pos = y_prob.ravel()
        try:
            auc = roc_auc_score(y_true, prob_pos) * 100.0
        except ValueError:
            auc = 0.0
    else:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro") * 100.0
        except ValueError:
            auc = 0.0
    # Binary: sensitivity = recall class 1, specificity = recall class 0.
    # zero_division=0: if that class has no samples in y_true, or model never predicts it, we get 0.
    if num_classes == 2:
        try:
            sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100.0
        except Exception:
            sensitivity = 0.0
        try:
            specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100.0
        except Exception:
            specificity = 0.0
    else:
        # Multi-class: sensitivity = macro-avg per-class recall (TP / (TP+FN));
        # specificity = macro-avg per-class TN/(TN+FP). Class with no samples in y_true contributes 0.0.
        try:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
            sens_list, spec_list = [], []
            for c in range(num_classes):
                tp = cm[c, c]
                fn = cm[c, :].sum() - tp
                fp = cm[:, c].sum() - tp
                tn = cm.sum() - tp - fn - fp
                sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
                spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
            sensitivity = float(np.mean(sens_list)) * 100.0
            specificity = float(np.mean(spec_list)) * 100.0
        except Exception:
            sensitivity = 0.0
            specificity = 0.0
    # F1: macro
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    return {
        "accuracy": acc,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
    }
