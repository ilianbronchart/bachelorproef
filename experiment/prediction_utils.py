import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from src.config import UNKNOWN_CLASS_ID
from torchvision.ops import box_iou

from experiment.settings import (
    CLASS_ID_TO_NAME,
    MISSING_GROUND_TRUTH_ID,
    MISSING_PREDICTION_CLASS_ID,
    SORTED_CLASS_IDS,
)


def create_confusion_matrix() -> pd.DataFrame:
    return pd.DataFrame(0, index=SORTED_CLASS_IDS, columns=SORTED_CLASS_IDS, dtype=int)


def update_confusion_matrix(
    confusion_mat: pd.DataFrame, eval_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Update an existing confusion matrix with results from eval_df.

    Parameters:
      confusion_mat: pd.DataFrame with index and columns = SORTED_CLASS_IDS plus special IDs
      eval_df: DataFrame returned by evaluate_predictions, containing 'true_class_id' and 'predicted_class_id'

    Returns:
      The updated confusion matrix (same object modified and returned).

    Note:
      - False positives (predictions with no matching GT) have true_class_id mapped to MISSING_GROUND_TRUTH_ID.
      - False negatives (GTs with no matching prediction) have predicted_class_id mapped to MISSING_PREDICTION_ID.
    """
    for _, row in eval_df.iterrows():
        true = row.get("true_class_id")
        pred = row.get("predicted_class_id")
        # Map false positives to MISSING_GROUND_TRUTH_ID
        t = MISSING_GROUND_TRUTH_ID if pd.isna(true) else int(true)
        # Map false negatives to MISSING_PREDICTION_ID
        p = MISSING_PREDICTION_CLASS_ID if pd.isna(pred) else int(pred)
        if t in confusion_mat.index and p in confusion_mat.columns:
            confusion_mat.loc[t, p] += 1
    return confusion_mat


def calculate_metrics(confusion_matrix: pd.DataFrame) -> dict[str, any]:
    """
    Calculate evaluation metrics from a confusion matrix.

    The function computes:
      - overall_accuracy: The ratio of correctly predicted ground truth instances (global TP)
        to the total number of instances.
      - micro (global) metrics: Precision, Recall, and F1 computed based on the global counts
        (global TP, FP, FN). This way, each instance is weighted equally.
      - per_class metrics: For each true class (row), computes Precision, Recall, F1, and Support.
      - known_accuracy: Accuracy computed only from the known predictions (i.e. excluding predictions
        labeled as unknown).

    Parameters:
        confusion_matrix (pd.DataFrame): Confusion matrix with rows as ground truth classes and
                                         columns as predicted classes (which may include unknown).

    Returns:
        dict[str, any]: Dictionary containing:
           * "overall_accuracy": Overall accuracy (global TP divided by total instances).
           * "micro": Global (micro-averaged) precision, recall, and F1.
           * "per_class": Per-class metrics.
           * "known_accuracy": Accuracy computed only on known predictions.
    """
    # Overall accuracy from all instances.
    total = confusion_matrix.to_numpy().sum()
    # Sum up true positives properly using label matching (not np.diag, which might be misaligned).
    overall_tp = sum(
        confusion_matrix.at[cls, cls]
        for cls in confusion_matrix.index
        if cls in confusion_matrix.columns
    )
    overall_accuracy = overall_tp / total if total > 0 else 0.0

    # Global (micro-averaged) counts.
    global_tp = 0
    global_fn = 0
    global_fp = 0

    per_class_metrics = {}

    for cls in confusion_matrix.index:
        # True positives (if there is a corresponding predicted column).
        tp = confusion_matrix.at[cls, cls] if cls in confusion_matrix.columns else 0
        # Support is the sum of the row.
        support = confusion_matrix.loc[cls].sum()
        # False negatives: ground truth instances of cls that were not predicted as cls.
        fn = support - tp
        # False positives: predictions of cls (column sum) minus true positives.
        fp = confusion_matrix[cls].sum() - tp if cls in confusion_matrix.columns else 0

        # unknown_rate: ratio of ground truth instances of cls that were predicted as unknown.
        unknown_rate = (
            (
                confusion_matrix.at[cls, UNKNOWN_CLASS_ID]
                if UNKNOWN_CLASS_ID in confusion_matrix.columns
                else 0
            )
            / support
            if support > 0
            else 0.0
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class_metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "unknown_rate": unknown_rate,
        }

        global_tp += tp
        global_fn += fn
        global_fp += fp

    micro_precision = (
        global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    )
    micro_recall = (
        global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    # Known-only accuracy: exclude the unknown column.
    known_columns = [
        col for col in confusion_matrix.columns if col not in [UNKNOWN_CLASS_ID]
    ]
    if known_columns:
        known_total = confusion_matrix[known_columns].to_numpy().sum()
        known_tp = sum(
            confusion_matrix.at[cls, cls]
            for cls in confusion_matrix.index
            if cls in known_columns
        )
        known_accuracy = known_tp / known_total if known_total > 0 else 0.0
    else:
        known_accuracy = None

    return {
        "overall_accuracy": overall_accuracy,
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },
        "per_class": per_class_metrics,
        "known_accuracy": known_accuracy,
    }


def render_confusion_matrix(cm: pd.DataFrame):
    """
    Render a confusion matrix using matplotlib.

    Parameters:
        cm (pd.DataFrame): Confusion matrix DataFrame with class IDs as both index (ground truth)
                           and columns (predicted).

    The function maps the class IDs to their names (if available) and displays the matrix as a heatmap.
    """
    # Map ground truth (rows) and predictions (columns) to class names.
    true_names = [CLASS_ID_TO_NAME.get(cid, str(cid)) for cid in cm.index]
    pred_names = [CLASS_ID_TO_NAME.get(cid, str(cid)) for cid in cm.columns]

    # Create the plot.
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    # Add a colorbar to provide scale reference.
    cbar = ax.figure.colorbar(im, ax=ax)

    # Set tick marks and labels using the mapped class names.
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_xticklabels(pred_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_yticklabels(true_names)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    # Annotate each cell with the count value.
    # Choose a contrasting text color based on the cell's value.
    thresh = cm.values.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cell_value = cm.iloc[i, j]
            color = "white" if cell_value > thresh else "black"
            ax.text(j, i, format(cell_value, "d"), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.show()


def create_per_class_metrics_df(grid_search_metrics):
    per_class_metrics_rows = []
    for grid_key, metrics in grid_search_metrics.items():
        labeling_set, sample_count, k, confidence, min_mask_area_size = grid_key

        for class_id, per_class_metric in metrics["per_class"].items():
            per_class_metrics_rows.append({
                "labeling_set": labeling_set,
                "sample_count": sample_count,
                "k": k,
                "confidence": confidence,
                "min_mask_area_size": min_mask_area_size,
                "class_id": class_id,
                **per_class_metric,
            })

    return pd.DataFrame(per_class_metrics_rows)


def evaluate_predictions(
    predictions_df: pd.DataFrame, gt_df: pd.DataFrame, iou_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Evaluate detection predictions against ground truth, labeling each prediction as:
      - TP: true positive (correct class match & IoU >= threshold)
      - FP: false positive (incorrect class or unmatched prediction)
      - TN: true negative (unknown-class prediction with no GT overlap)
      - FN: false negative (ground truth with no matching prediction) as MISSING_PREDICTION_CLASS_ID

    Returns a DataFrame combining prediction rows (with new 'label', 'true_class_id', 'predicted_class_id')
    and additional FN rows for unmatched ground truths.

    predictions_df must include:
      ['frame_idx','object_id','confidence','mask_area',
       'x1','y1','x2','y2','predicted_class_id','predicted_confidence']

    gt_df must include:
      ['frame_idx','class_id','mask_area','laplacian_variance','x1','y1','x2','y2']
    """
    records = []

    preds = predictions_df.reset_index(drop=True)
    gts = gt_df.reset_index(drop=True)

    all_frames = sorted(set(preds["frame_idx"]).union(gts["frame_idx"]))
    for frame in all_frames:
        preds_f = preds[preds["frame_idx"] == frame].reset_index(drop=True)
        gts_f = gts[gts["frame_idx"] == frame].reset_index(drop=True)

        if len(preds_f) and len(gts_f):
            # compute IoU
            boxes_preds = torch.tensor(
                preds_f[["x1", "y1", "x2", "y2"]].values, dtype=torch.float
            )
            boxes_gts = torch.tensor(
                gts_f[["x1", "y1", "x2", "y2"]].values, dtype=torch.float
            )
            iou_mat = box_iou(boxes_preds, boxes_gts).cpu().numpy()

            # greedy matching
            pairs = [
                (i, j, iou_mat[i, j])
                for i in range(iou_mat.shape[0])
                for j in range(iou_mat.shape[1])
                if iou_mat[i, j] >= iou_threshold
            ]
            pairs.sort(key=lambda x: -x[2])

            matched_preds, matched_gts = set(), set()
            for i, j, _ in pairs:
                if i in matched_preds or j in matched_gts:
                    continue
                matched_preds.add(i)
                matched_gts.add(j)
                p = preds_f.iloc[i].to_dict()
                p["true_class_id"] = gts_f.iloc[j]["class_id"]
                p["label"] = (
                    "TP" if p["predicted_class_id"] == p["true_class_id"] else "FP"
                )
                records.append(p)

            # unmatched predictions
            for i in range(len(preds_f)):
                if i in matched_preds:
                    continue
                p = preds_f.iloc[i].to_dict()
                p["true_class_id"] = np.nan
                p["label"] = "TN" if p["predicted_class_id"] == UNKNOWN_CLASS_ID else "FP"
                records.append(p)

            # unmatched ground truths -> FN with MISSING_PREDICTION_CLASS_ID
            for j in range(len(gts_f)):
                if j in matched_gts:
                    continue
                gt = gts_f.iloc[j]
                fn = {col: np.nan for col in predictions_df.columns}
                for col in ["frame_idx", "x1", "y1", "x2", "y2", "mask_area"]:
                    fn[col] = gt[col]
                fn["predicted_class_id"] = MISSING_PREDICTION_CLASS_ID
                fn["predicted_confidence"] = 0.0
                fn["true_class_id"] = gt["class_id"]
                fn["label"] = "FN"
                records.append(fn)

        elif len(preds_f):  # no ground truths
            for i in range(len(preds_f)):
                p = preds_f.iloc[i].to_dict()
                p["true_class_id"] = np.nan
                p["label"] = "TN" if p["predicted_class_id"] == UNKNOWN_CLASS_ID else "FP"
                records.append(p)

        else:  # no predictions
            for _, gt in gts_f.iterrows():
                fn = {col: np.nan for col in predictions_df.columns}
                for col in ["frame_idx", "x1", "y1", "x2", "y2", "mask_area"]:
                    fn[col] = gt[col]
                fn["predicted_class_id"] = MISSING_PREDICTION_CLASS_ID
                fn["predicted_confidence"] = 0.0
                fn["true_class_id"] = gt["class_id"]
                fn["label"] = "FN"
                records.append(fn)

    eval_df = pd.DataFrame(records)
    return eval_df
