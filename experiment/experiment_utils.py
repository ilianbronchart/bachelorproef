from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from src.config import UNKNOWN_CLASS_ID
from torchvision.ops import box_iou

from experiment.settings import (
    CLASS_ID_TO_NAME,
    MISSING_PREDICTION_CLASS_ID,
    SORTED_CLASS_IDS,
    OBJECT_DATASETS_PATH,
    GROUND_TRUTH_PATH,
    MISSING_GROUND_TRUTH_CLASS_ID
)

def get_object_df(
    recording_id: str,
    drop_embedding: bool = False,
) -> pd.DataFrame:
    object_df_path = OBJECT_DATASETS_PATH / f"{recording_id}.csv"
    if not object_df_path.exists():
        raise ValueError(
            f"Object df path {object_df_path} does not exist. Please run the object detection pipeline first."
        )
    object_df = pd.read_csv(object_df_path)

    if drop_embedding:
        object_df = object_df.drop(columns=["embedding"])
    
    return object_df

def get_ground_truth_df(ignored_class_ids: list[int]) -> pd.DataFrame:
    if not GROUND_TRUTH_PATH.exists():
        raise ValueError(
            f"Ground truth df path {GROUND_TRUTH_PATH} does not exist. Please run the ground truth pipeline first."
        )
    gt_df = pd.read_csv(GROUND_TRUTH_PATH)

    # drop rows where class_id is in ignored_class_ids
    gt_df = gt_df[~gt_df["class_id"].isin(ignored_class_ids)]
    
    return gt_df

def create_confusion_matrix(ignored_class_ids: list[int]) -> pd.DataFrame:
    class_ids = [
        class_id
        for class_id in SORTED_CLASS_IDS
        if class_id not in ignored_class_ids and class_id != UNKNOWN_CLASS_ID
    ]
    return pd.DataFrame(0, index=class_ids, columns=class_ids, dtype=int)


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
    """

    for _, row in eval_df.iterrows():
        true = row.get("true_class_id")
        pred = row.get("predicted_class_id")

        if pd.isna(true):
            if pred == UNKNOWN_CLASS_ID:
                # This is a true negative (TN) for the unknown class.
                continue

            # This is a false positive (FP) for the predicted class.
            true = MISSING_GROUND_TRUTH_CLASS_ID

        t = int(true)
        p = int(pred)
        if t in confusion_mat.index and p in confusion_mat.columns:
            # This is a true positive (TP) for the predicted class.
            confusion_mat.loc[t, p] += 1

@dataclass
class ClassMetrics:
    class_id: int
    precision: float
    recall: float
    f1: float
    support: int

@dataclass
class CMMetrics:
    overall_accuracy: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    per_class_metrics: list[ClassMetrics]

def confusion_matrix_metrics(confusion_matrix: pd.DataFrame) -> dict[str, any]:
    """
    Calculate evaluation metrics from a confusion matrix.

    The function computes:
      - overall_accuracy: The ratio of correctly predicted ground truth instances (global TP)
        to the total number of instances.
      - micro (global) metrics: Precision, Recall, and F1 computed based on the global counts
        (global TP, FP, FN). This way, each instance is weighted equally.
      - per_class metrics: For each true class (row), computes Precision, Recall, F1, and Support.

    Parameters:
        confusion_matrix (pd.DataFrame): Confusion matrix with rows as ground truth classes and
                                         columns as predicted classes.

    Returns: CMMetrics
    """

    total = confusion_matrix.to_numpy().sum()
    overall_tp = np.diag(confusion_matrix).sum()
    overall_accuracy = overall_tp / total if total > 0 else 0.0

    # Global (micro-averaged) counts.
    global_tp = 0
    global_fn = 0
    global_fp = 0

    per_class_metrics = []
    for cls in confusion_matrix.index:
        if cls < 0:
            # skip special classes
            continue

        # True positives (if there is a corresponding predicted column).
        tp = confusion_matrix.at[cls, cls] if cls in confusion_matrix.columns else 0
        # Support is the sum of the row.
        support = confusion_matrix.loc[cls].sum()
        # False negatives: ground truth instances of cls that were not predicted as cls.
        fn = support - tp
        # False positives: predictions of cls (column sum) minus true positives.
        fp = confusion_matrix[cls].sum() - tp if cls in confusion_matrix.columns else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class_metrics.append(
            ClassMetrics(
                class_id=cls,
                precision=precision,
                recall=recall,
                f1=f1,
                support=support,
            )
        )

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

    return CMMetrics(
        overall_accuracy=overall_accuracy,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
        per_class_metrics=per_class_metrics,
    )

def print_confusion_matrix_metrics(metrics: CMMetrics):
    """
    Print confusion matrix metrics in a readable format.

    Parameters:
        metrics (CMMetrics): The metrics object containing overall and per-class metrics.
    """
    print(f"Overall Accuracy: {metrics.overall_accuracy:.4f}")
    print(f"Micro Precision: {metrics.micro_precision:.4f}")
    print(f"Micro Recall: {metrics.micro_recall:.4f}")
    print(f"Micro F1: {metrics.micro_f1:.4f}\n")

    print("Per Class Metrics:")
    # print in table format
    print(f"{'Class Name':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    for class_metric in metrics.per_class_metrics:
        print(
            f"{CLASS_ID_TO_NAME[class_metric.class_id]:<10} {class_metric.precision:<10.4f} "
            f"{class_metric.recall:<10.4f} {class_metric.f1:<10.4f} "
            f"{class_metric.support:<10}"
        )

def render_confusion_matrix(cm: pd.DataFrame, show_absolute_counts: bool = False):
    """
    Render a confusion matrix using matplotlib.

    Parameters:
        cm (pd.DataFrame): Confusion matrix DataFrame with class IDs as both index (ground truth)
                           and columns (predicted).

    The function maps the class IDs to their names (if available) and displays the matrix as a heatmap.
    """
    # Ensure the confusion matrix is square and has class IDs as both index and columns.
    if cm.shape[0] != cm.shape[1]:
        raise ValueError(
            "Confusion matrix must be square with class IDs as both index and columns."
        )


    # if show_absolute_counts is false, normalize the confusion matrix
    if not show_absolute_counts:
        cm = cm.div(cm.sum(axis=1), axis=0).fillna(0)

    # Map ground truth (rows) and predictions (columns) to class names.
    true_names = [CLASS_ID_TO_NAME.get(cid, str(cid)) for cid in cm.index]
    pred_names = [CLASS_ID_TO_NAME.get(cid, str(cid)) for cid in cm.columns]

    # Create the plot.
    figsize_per_cell = 0.6  # Adjust if needed
    rows, cols = cm.shape
    fig, ax = plt.subplots(figsize=(cols * figsize_per_cell, rows * figsize_per_cell))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

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
            
            # Handle zero values as integers
            if cell_value == 0:
                text = "0"
            else:
                format_str = ".2f" if not show_absolute_counts else "d" if isinstance(cell_value, (int, np.integer)) else ".2f"
                text = format(cell_value, format_str)
                
            ax.text(j, i, text, ha="center", va="center", color=color)

    plt.tight_layout()
    plt.show()

def evaluate_predictions(
    predictions_df: pd.DataFrame, gt_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Evaluate detection predictions against ground truth
    Labels each prediction as:
      - TP: true positive (predicted_class_id matches a true_class_id from an available GT
            in the same frame, assigned greedily by prediction confidence).
      - FP: false positive (unmatched prediction, not UNKNOWN_CLASS_ID).
      - TN: true negative (unmatched prediction of UNKNOWN_CLASS_ID).
      - FN: false negative (ground truth with no matching prediction).

    Returns a DataFrame combining prediction rows (with new 'label', 'true_class_id')
    and additional FN rows for unmatched ground truths.

    predictions_df must include:
      ['frame_idx', 'predicted_class_id', 'predicted_confidence']
      (Other columns like 'object_id', 'x1', 'y1', 'x2', 'y2' are preserved if present)

    gt_df must include:
      ['frame_idx', 'class_id']
      (Other columns like 'x1', 'y1', 'x2', 'y2' are used for FN records if present)
    """
    records = []

    preds = predictions_df.reset_index(drop=True)
    gts = gt_df.reset_index(drop=True)

    # Determine all unique frames present in either predictions or ground truths
    pred_frames = set(preds["frame_idx"]) if not preds.empty else set()
    gt_frames = set(gts["frame_idx"]) if not gts.empty else set()
    all_frames = sorted(list(pred_frames.union(gt_frames)))

    for frame in all_frames:
        preds_f = preds[preds["frame_idx"] == frame].reset_index(drop=True)
        gts_f = gts[gts["frame_idx"] == frame].reset_index(drop=True)

        if len(preds_f) > 0 and len(gts_f) > 0:
            # Create potential pairs based on class match
            potential_pairs = []  # Stores (pred_confidence, pred_idx, gt_idx)
            for i in range(len(preds_f)):
                pred_class = preds_f.iloc[i]["predicted_class_id"]
                # Ensure 'predicted_confidence' column exists and is used
                pred_conf = preds_f.iloc[i].get("predicted_confidence", preds_f.iloc[i].get("confidence", 0.0))

                for j in range(len(gts_f)):
                    gt_class = gts_f.iloc[j]["class_id"]
                    if pred_class == gt_class:
                        potential_pairs.append((pred_conf, i, j))

            # Sort potential pairs by prediction confidence (descending)
            potential_pairs.sort(key=lambda x: x[0], reverse=True)

            matched_pred_indices = set()
            matched_gt_indices = set()

            # Greedy assignment for TP
            for pred_conf, pred_idx, gt_idx in potential_pairs:
                if pred_idx in matched_pred_indices or gt_idx in matched_gt_indices:
                    continue  # This prediction or GT is already matched

                matched_pred_indices.add(pred_idx)
                matched_gt_indices.add(gt_idx)

                p_dict = preds_f.iloc[pred_idx].to_dict()
                p_dict["true_class_id"] = gts_f.iloc[gt_idx]["class_id"]
                # Since pairs are formed on class equality, this is always a TP
                p_dict["label"] = "TP"
                records.append(p_dict)

            # Handle unmatched predictions (become FP or TN)
            for i in range(len(preds_f)):
                if i not in matched_pred_indices:
                    p_dict = preds_f.iloc[i].to_dict()
                    p_dict["true_class_id"] = np.nan  # No matched GT
                    if p_dict["predicted_class_id"] == UNKNOWN_CLASS_ID:
                        p_dict["label"] = "TN"
                    else:
                        p_dict["true_class_id"] = MISSING_GROUND_TRUTH_CLASS_ID
                        p_dict["label"] = "FP"
                    records.append(p_dict)

            # Handle unmatched ground truths (become FN)
            for j in range(len(gts_f)):
                if j not in matched_gt_indices:
                    gt_row = gts_f.iloc[j]
                    # Create a base FN record with columns from predictions_df for consistency
                    fn_record = {col: np.nan for col in predictions_df.columns}

                    fn_record["frame_idx"] = gt_row["frame_idx"]
                    fn_record["true_class_id"] = gt_row["class_id"]
                    fn_record["predicted_class_id"] = MISSING_PREDICTION_CLASS_ID
                    fn_record["predicted_confidence"] = 0.0 # Default for missing preds
                    fn_record["label"] = "FN"

                    # Carry over spatial info or other GT columns if present
                    for col in gt_df.columns:
                        if col not in fn_record or pd.isna(fn_record[col]): # Prioritize already set values
                             if col in gt_row:
                                fn_record[col] = gt_row[col]
                    records.append(fn_record)

        elif len(preds_f) > 0:  # Only predictions, no ground truths in this frame
            for i in range(len(preds_f)):
                p_dict = preds_f.iloc[i].to_dict()
                p_dict["true_class_id"] = np.nan
                if p_dict["predicted_class_id"] == UNKNOWN_CLASS_ID:
                    p_dict["label"] = "TN"
                else:
                    p_dict["true_class_id"] = MISSING_GROUND_TRUTH_CLASS_ID
                    p_dict["label"] = "FP"
                records.append(p_dict)

        elif len(gts_f) > 0:  # Only ground truths, no predictions in this frame
            for j in range(len(gts_f)):
                gt_row = gts_f.iloc[j]
                fn_record = {col: np.nan for col in predictions_df.columns}

                fn_record["frame_idx"] = gt_row["frame_idx"]
                fn_record["true_class_id"] = gt_row["class_id"]
                fn_record["mask_area"] = gt_row["mask_area"]
                fn_record["predicted_class_id"] = MISSING_PREDICTION_CLASS_ID
                fn_record["predicted_confidence"] = 0.0
                fn_record["label"] = "FN"

                for col in gt_df.columns:
                    if col not in fn_record or pd.isna(fn_record[col]):
                        if col in gt_row:
                            fn_record[col] = gt_row[col]
                records.append(fn_record)
        # If both preds_f and gts_f are empty for a frame, nothing happens for this frame.

    eval_df = pd.DataFrame(records) if records else pd.DataFrame(columns=list(predictions_df.columns) + ['label', 'true_class_id'])
    
    # Ensure standard output columns exist, even if records list was empty
    expected_cols = list(predictions_df.columns)
    if 'label' not in expected_cols: expected_cols.append('label')
    if 'true_class_id' not in expected_cols: expected_cols.append('true_class_id')

    # Add missing expected columns with NaN if they don't exist after DataFrame creation
    for col in expected_cols:
        if col not in eval_df.columns:
            eval_df[col] = np.nan

    # Raise an error if any true class or predicted class is nan
    pred_class_missing = eval_df["predicted_class_id"].isna()
    if pred_class_missing.any():
        raise ValueError(
            f"Some predicted class IDs are missing in the evaluation DataFrame:\n{eval_df[pred_class_missing]}"
        )

    return eval_df