import scipy.stats as stats
from sklearn.metrics import (
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score,
    auc,
    confusion_matrix,
    balanced_accuracy_score,
)
import numpy as np
import numpy as np
from copy import deepcopy
import torch
import pandas as pd


def kendal_rank_correlation(predictions, labels):
    if torch.is_tensor(predictions):
        predictions_np = predictions.cpu().numpy().flatten()
    else:
        predictions_np = np.array(predictions)
    if torch.is_tensor(labels):
        labels_np = labels.cpu().numpy().flatten()
    else:
        labels_np = np.array(labels)
    tau, _ = stats.kendalltau(predictions_np, labels_np)

    return tau


def class_absolute_error(predictions, labels):
    """Gets the average mean absolute error"""
    predictions_np = (
        predictions.cpu().numpy().flatten() if torch.is_tensor(predictions) else np.array(predictions)
    )
    labels_np = labels.cpu().numpy().flatten() if torch.is_tensor(labels) else np.array(labels)

    label_set = np.unique(labels_np)
    all_mae = []
    for label in label_set:
        index_list = np.where(labels_np == label)[0]
        pred_list = predictions_np[index_list]
        label_list = labels_np[index_list]
        mae = mean_absolute_error(label_list, pred_list)
        all_mae.append(mae)

    return np.average(all_mae)


def get_precision_recall_f1_extreme_bins(predictions, labels, positive_pred_bins, positive_label_bins):
    # Ensure predictions and labels are on CPU and converted to Python lists if they are tensors
    predictions_np = (
        predictions.cpu().numpy().flatten() if torch.is_tensor(predictions) else np.array(predictions)
    )
    labels_np = labels.cpu().numpy().flatten() if torch.is_tensor(labels) else np.array(labels)

    # Convert bins to sets for faster lookup
    positive_pred_bins_set = set(positive_pred_bins)
    positive_label_bins_set = set(positive_label_bins)

    # Generate binary labels based on bin membership
    all_preds = ["t" if pred in positive_pred_bins_set else "o" for pred in predictions_np]
    all_labels = ["t" if label in positive_label_bins_set else "o" for label in labels_np]

    # Compute precision, recall, and F1 score
    precision = precision_score(
        y_true=all_labels,
        y_pred=all_preds,
        average="binary",
        pos_label="t",
        zero_division=0,
    )
    recall = recall_score(
        y_true=all_labels,
        y_pred=all_preds,
        average="binary",
        pos_label="t",
        zero_division=0,
    )
    f1 = f1_score(
        y_true=all_labels,
        y_pred=all_preds,
        average="binary",
        pos_label="t",
        zero_division=0,
    )

    return precision, recall, f1


def imbalanced_ordinal_classification_index(conf_mat, beta=None, missing="zeros", verbose=False):
    # missing: 'zeros', 'uniform', 'diagonal'

    N = int(np.sum(conf_mat))
    K = float(conf_mat.shape[0])
    gamma = 1.0
    if beta is None:
        beta_vals = np.linspace(0.0, 1.0, 1000).transpose()
    else:
        beta_vals = [beta]

    # Fixing missing classes
    conf_mat_fixed = deepcopy(conf_mat)
    for ii in range(conf_mat.shape[0]):
        if np.sum(conf_mat[ii, :]) == 0:
            if missing == "zeros":
                K -= 1.0  # Dealt with by 0**Nr[rr]
            elif missing == "uniform":
                conf_mat_fixed[ii, :] = np.ones((1, conf_mat.shape[1]))
            elif missing == "diagonal":
                conf_mat_fixed[ii, ii] = 1
            else:
                raise ValueError("Unknown way of dealing with missing classes.")

    # Computing number of samples in each class
    Nr = np.sum(conf_mat_fixed, axis=1)

    beta_oc = list()

    # Computing total dispersion and helper matrices
    helper_mat2 = np.zeros(conf_mat_fixed.shape)
    for rr in range(conf_mat_fixed.shape[0]):
        for cc in range(conf_mat_fixed.shape[1]):
            helper_mat2[rr, cc] = (
                float(conf_mat_fixed[rr, cc]) / (Nr[rr] + 0 ** Nr[rr]) * ((abs(rr - cc)) ** gamma)
            )
    total_dispersion = np.sum(helper_mat2) ** (1 / gamma)
    helper_mat1 = np.zeros(conf_mat_fixed.shape)
    for rr in range(conf_mat_fixed.shape[0]):
        for cc in range(conf_mat_fixed.shape[1]):
            helper_mat1[rr, cc] = float(conf_mat_fixed[rr, cc]) / (Nr[rr] + 0 ** Nr[rr])
    helper_mat1 = np.divide(helper_mat1, total_dispersion + K)

    for beta in beta_vals:
        beta = beta / K

        # Creating error matrix and filling first entry
        error_mat = np.zeros(conf_mat_fixed.shape)
        error_mat[0, 0] = 1 - helper_mat1[0, 0] + beta * helper_mat2[0, 0]

        # Filling column 0
        for rr in range(1, conf_mat_fixed.shape[0]):
            cc = 0
            error_mat[rr, cc] = error_mat[rr - 1, cc] - helper_mat1[rr, cc] + beta * helper_mat2[rr, cc]

        # Filling row 0
        for cc in range(1, conf_mat_fixed.shape[1]):
            rr = 0
            error_mat[rr, cc] = error_mat[rr, cc - 1] - helper_mat1[rr, cc] + beta * helper_mat2[rr, cc]

        # Filling the rest of the error matrix
        for cc in range(1, conf_mat_fixed.shape[1]):
            for rr in range(1, conf_mat_fixed.shape[0]):
                cost_up = error_mat[rr - 1, cc]
                cost_left = error_mat[rr, cc - 1]
                cost_lefttop = error_mat[rr - 1, cc - 1]
                aux = np.min([cost_up, cost_left, cost_lefttop])
                error_mat[rr, cc] = aux - helper_mat1[rr, cc] + beta * helper_mat2[rr, cc]

        beta_oc.append(error_mat[-1, -1])

    if len(beta_vals) < 2:
        return beta_oc[0]
    else:
        # if verbose:
        # plot_uoc(beta_vals, beta_oc)
        return auc(beta_vals, beta_oc)


def wilson_index(predictions, labels):
    predictions_np = (
        predictions.cpu().numpy().flatten() if torch.is_tensor(predictions) else np.array(predictions)
    )
    labels_np = labels.cpu().numpy().flatten() if torch.is_tensor(labels) else np.array(labels)
    return imbalanced_ordinal_classification_index(confusion_matrix(labels_np, predictions_np))


def get_balanced_accuracy(predictions, labels, step):
    predictions_np = (
        predictions.cpu().numpy().flatten() if torch.is_tensor(predictions) else np.array(predictions)
    )
    labels_np = labels.cpu().numpy().flatten() if torch.is_tensor(labels) else np.array(labels)
    return balanced_accuracy_score(labels_np, predictions_np)


def get_metric(metric, all_preds, all_labels):
    if metric == "kendall":
        return kendal_rank_correlation(all_preds, all_labels)
    elif metric == "amae":
        return class_absolute_error(all_preds, all_labels)
    elif metric == "lowbinf1":
        return get_precision_recall_f1_extreme_bins(
            all_preds, all_labels, positive_pred_bins=[1, 2], positive_label_bins=[1, 2]
        )[-1]
    elif metric == "highbinf1":
        return get_precision_recall_f1_extreme_bins(
            all_preds, all_labels, positive_pred_bins=[7, 8], positive_label_bins=[7, 8]
        )[-1]
    elif metric == "wilson_index":
        return wilson_index(all_preds, all_labels)


def get_table_metric(df, metric, if_highlight):
    table = [
        [
            "",
            "Expert_1",
            "Expert_2",
            "Expert_3",
            "Expert_4",
            "Expert_5",
            "One-hot",
        ]
    ]

    # rows to predict median and each radiologist
    for label_column in [
        "GT-Median",
        "Expert_1",
        "Expert_2",
        "Expert_3",
        "Expert_4",
        "Expert_5",
    ]:
        pred_temp_list = [label_column]
        if label_column == "GT-Median":
            label_column = "Label"
        # radiologists
        for column in ["Expert_1", "Expert_2", "Expert_3", "Expert_4", "Expert_5"]:
            label_list = df[label_column].tolist()
            pred_list = df[column].tolist()
            pred_temp_list.append(
                get_metric(
                    metric,
                    pred_list,
                    label_list,
                )
            )
        # models

        label_list = df["Label"].tolist()
        pred_list = df["Prediction"].tolist()
        pred_temp_list.append(
            get_metric(
                metric,
                pred_list,
                label_list,
            )
        )

        table.append(pred_temp_list)

    table_df = pd.DataFrame(table[1:], columns=table[0])

    for i in range(1, 6):
        table_df["Expert_" + str(i)] = table_df["Expert_" + str(i)].apply(float)

    table_df = table_df.apply(pd.to_numeric, errors="ignore")
    table_df = table_df.round(4)

    if not if_highlight:
        return table_df
    else:
        if metric == "amae":
            table_df = table_df.replace(0, np.nan)
            return table_df.style.highlight_min(color="lightgreen", axis=1)
        else:
            table_df = table_df.replace(1, np.nan)
            return table_df.style.highlight_max(color="lightgreen", axis=1)
