import os
import pandas as pd
from tqdm import tqdm
import argparse
from collections import Counter
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    classification_report,
)

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# from networks.networks import efficientb0
from dataset.get_dataset import prepare_transforms
from utils.utils import get_criterion, set_seed, get_num_classes
from dataset.dataset import CSAWM
from networks.networks import ResNet
from val import evaluate_fn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import re


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CSAWM", help="dataset name")
    parser.add_argument("--exp", type=str, default="", help="experiment name")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    parser.add_argument("--mode", type=str, default="test", help="test or valid")
    parser.add_argument(
        "--deterministic",
        type=int,
        default=1,
        help="whether use deterministic training",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        help="Loss to use: ce, ordinal_encoding, binomial_unimodal",
    )
    args = parser.parse_args()
    return args


def compute_class_based_similarity(
    embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int
):
    class_embeddings = []

    for class_idx in range(num_classes):
        # Extract embeddings for samples belonging to the current class
        class_embeddings_for_samples = embeddings[labels == class_idx]

        # Compute the mean embedding for the current class
        class_average_embedding = class_embeddings_for_samples.mean(dim=0)
        class_embeddings.append(class_average_embedding)

    # Stack the class-wise embeddings into a tensor
    class_embeddings = torch.stack(class_embeddings)

    # Normalize the embeddings for cosine similarity
    class_embeddings_normalized = F.normalize(class_embeddings, p=2, dim=1)

    # Compute cosine similarity between each pair of classes
    similarity_matrix = torch.mm(
        class_embeddings_normalized, class_embeddings_normalized.T
    )

    return similarity_matrix


def save_plot_in_model_folder(
    ckpt_path: Path, model_id: str, plot: plt, plot_filename: str
):
    """Save a plot in a folder named after the model inside the test_results folder."""
    # Extract the parent directory three levels above the checkpoint file
    ckpt_path = Path(ckpt_path)
    parent_dir = ckpt_path.parents[3]

    # Create the test_results directory in the parent folder
    results_dir = parent_dir / "test_results"

    # Create the model_name directory inside the test_results directory
    model_dir = results_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save the plot as a PDF in the model_name directory
    plot_path = model_dir / f"{plot_filename}.pdf"
    plot.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")


def plot_confusion_matrix(
    cm, class_names, ckpt_path, model_id, filename="confusion_matrix"
):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if cm.dtype.kind == "f" else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    save_plot_in_model_folder(ckpt_path, model_id, plt, filename)
    plt.close()


def plot_class_similarity(
    class_similarity_matrix: torch.Tensor, ckpt_path: Path, model_id: str
):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        class_similarity_matrix.cpu().detach().numpy(),
        cmap="Blues",
        annot=True,
        fmt=".2f",
    )
    plt.title("Class-Based Embedding Similarity Matrix")
    save_plot_in_model_folder(ckpt_path, model_id, plt, "class_embedding_similarity")


if __name__ == "__main__":
    args = args_parser()
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ------------------------------ deterministic or not ------------------------------
    cudnn.benchmark = False
    cudnn.deterministic = True
    set_seed(args)

    # ------------------------------ dataset and dataloader ------------------------------
    csv_test = "/processing/v.corbetta/CSAW-M/labels/CSAW-M_test.csv"
    test_data_dir = "/processing/v.corbetta/CSAW-M/images/preprocessed/test"
    csv_df = pd.read_csv(csv_test, sep=";")
    # test_df = csv_df[csv_df[["Filename"], ["Label"]]]
    csv_df["Label"] = csv_df["Label"] - 1
    _, test_aug = prepare_transforms()
    test_dataset = CSAWM(csv_df, test_data_dir, 8, test_aug, args.loss)
    # _, _, test_dataset = get_datasets(args)
    print(Counter(test_dataset.targets))
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    # args.loss = "ce"
    args.num_classes = 8

    # ------------------------------ load model ------------------------------
    # model = efficientb0(n_classes=test_dataset.n_classes).cuda()
    model_dir = "/projects/federated_ordinal_classification/results/fedala_fediic_parameters_study_ordinal_encoding/adam_1e-05_5_20_80_2_16/print_model/CSAWM_0.9_0.8_ordinal_encoding/0/train/checkpoints/best_model.ckpt"
    pattern = r"(ce|ordinal_encoding|binomial)"
    match = re.findall(pattern, model_dir)
    print(match)
    if match:
        args.loss = match[-1]
    print(f"{args.loss=}")
    num_classes = get_num_classes(args.loss, args.num_classes)
    print(f"{args.num_classes=}")
    model = ResNet(args=args)
    model.load_state_dict(torch.load(model_dir))
    model.to(device)

    print("have loaded the best model from {}".format(model_dir))

    # ------------------------------ test or valid ------------------------------
    np.set_printoptions(suppress=True)

    all_preds = []
    all_labels = []
    all_prob = []
    all_embeddings = []
    model.eval()

    for x, label in tqdm(test_loader):
        x = x.to(device)
        label = label.to(device)
        batch_embeddings = []

        def hook_fn(module, input, output, batch_embeddings):
            batch_embeddings.append(output.cpu().view(output.size(0), -1))

        hook = model.model.avgpool.register_forward_hook(
            lambda module, input, output: hook_fn(
                module, input, output, batch_embeddings
            )
        )

        with torch.no_grad():
            _, logits = model(x)
        hook.remove()
        batch_embeddings = torch.cat(batch_embeddings, dim=0)
        all_embeddings.append(batch_embeddings.cpu())
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)

        all_prob.append(prob.cpu())
        all_preds.append(pred.cpu())
        all_labels.append(label)
        model.zero_grad()

    all_prob = torch.cat(all_prob).numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_embeddings = torch.cat(all_embeddings, dim=0)
    class_similarity_matrix = compute_class_based_similarity(
        all_embeddings, all_labels, num_classes
    )
    plot_class_similarity(class_similarity_matrix, model_dir, model_id="0.9_0.8_0")

    metrics = evaluate_fn(model, test_loader, device, args.loss, num_classes)
    print(
        f"amae: {metrics['amae']}, wilson_idx: {metrics['wilson_idx']}, kendall's tau: {metrics['kendall_tau']}, balanced_accuracy: {metrics['balanced_accuracy']}"
    )

    conf_matrix = metrics["confusion_matrix"]
    print(conf_matrix)
    class_names = [f"Class {i}" for i in range(num_classes)]
    plot_confusion_matrix(conf_matrix, class_names, model_dir, model_id="0.9_0.8_0")
