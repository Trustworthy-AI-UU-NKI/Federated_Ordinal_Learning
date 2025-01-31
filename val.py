import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from tqdm import tqdm

from utils.utils import get_criterion
from utils.metrics import (
    class_absolute_error,
    kendal_rank_correlation,
    wilson_index,
)


def compute_bacc(model, dataloader, get_confusion_matrix, args):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for x, label in dataloader:
            x = x.cuda()
            _, logits = model(x)
            pred = torch.argmax(logits, dim=1)

            all_preds.append(pred.cpu())
            all_labels.append(label)

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    acc = balanced_accuracy_score(all_labels, all_preds)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(all_labels, all_preds)

    if get_confusion_matrix:
        return acc, conf_matrix
    else:
        return acc


@torch.no_grad()
def evaluate_fn(model, dataloader, device, criterion_name, num_classes):
    model.eval()
    criterion = get_criterion(criterion_name, num_classes, device)
    preds_eval, labels_eval = [], []
    for images, labels in tqdm(dataloader):
        images, labels = (
            images.to(device),
            labels.to(device),
        )
        _, outputs = model(images)
        outputs.to(device)
        # NOTE: criterion could be `loss_fn`
        if criterion_name == "ordinal_encoding":
            _, preds = criterion(outputs, labels)
        else:
            preds = torch.argmax(outputs, dim=1)

        labels_eval.append(labels)
        preds_eval.append(preds)

    # Stack evaluation results
    # all_preds = torch.stack(preds_eval)
    # all_labels = torch.stack(labels_eval)

    all_labels = torch.cat(labels_eval).cpu().numpy()
    all_preds = torch.cat(preds_eval).cpu().numpy()

    # Compute metrics
    amae = class_absolute_error(all_preds, all_labels)
    wilson_idx = wilson_index(all_preds, all_labels)
    kendalls_tau = kendal_rank_correlation(all_preds, all_labels)
    bal_accuracy = balanced_accuracy_score(all_labels, all_preds)
    confusion_mat = confusion_matrix(all_labels, all_preds)
    return {
        "amae": amae,
        "wilson_idx": wilson_idx,
        "kendall_tau": kendalls_tau,
        "balanced_accuracy": bal_accuracy,
        "confusion_matrix": confusion_mat,
    }


def compute_loss(model, dataloader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for x, label in dataloader:
            if isinstance(x, list):
                x = x[0]
            x, label = x.to(device), label.to(device)
            _, logits = model(x)
            loss += criterion(logits, label)
    return loss


def compute_loss_of_classes_(model, dataloader, n_classes, device, loss_name):
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.eval()

    loss_class = torch.zeros(n_classes).float()
    loss_list = []
    label_list = []

    with torch.no_grad():
        for x, label in dataloader:
            if isinstance(x, list):
                x = x[0]
            x, label = x.to(device), label.to(device)
            _, logits = model(x)
            loss = criterion(logits, label)
            loss_list.append(loss)
            label_list.append(label)

    loss_list = torch.cat(loss_list).cpu()
    label_list = torch.cat(label_list).cpu()

    for i in range(n_classes):
        idx = torch.where(label_list == i)[0]
        loss_class[i] = loss_list[idx].sum()

    return loss_class


def compute_loss_of_classes(model, dataloader, n_classes, device, loss_name):
    # criterion = nn.CrossEntropyLoss(reduction="none")
    criterion = get_criterion(loss_name, n_classes, device, per_class=True).to(device)
    model.eval()

    loss_class = torch.zeros(n_classes).float()
    loss_list = []
    label_list = []

    with torch.no_grad():
        for x, label in dataloader:
            if isinstance(x, list):
                x = x[0]
            x, label = x.to(device), label.to(device)
            _, logits = model(x)
            if (
                criterion.__class__.__name__ == "OrdinalEncodingLoss"
            ):  # Adjust for your loss function name
                loss, _ = criterion(logits, label, per_class=True)
            else:
                loss = criterion(logits, label)
            loss_list.append(loss)
            label_list.append(label)

    loss_list = torch.cat(loss_list).cpu()
    label_list = torch.cat(label_list).cpu()

    for i in range(n_classes):
        idx = torch.where(label_list == i)[0]
        loss_class[i] = loss_list[idx].sum()

    return loss_class
