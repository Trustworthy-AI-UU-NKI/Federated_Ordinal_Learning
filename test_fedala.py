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
from utils.utils import set_seed, get_num_classes
from dataset.dataset import CSAWM
from networks.networks import ResNet
from val import evaluate_fn
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

    model_dir = "/projects/federated_ordinal_classification/results/fedala_fediic_parameters_study_ordinal_encoding/adam_1e-05_5_20_80_2_16/print_model/CSAWM_0.9_0.8_ordinal_encoding/3/train/checkpoints/best_model.ckpt"
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
    model.eval()
    with torch.no_grad():
        for x, label in tqdm(test_loader):
            x = x.to(device)
            _, logits = model(x)
            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)

            all_prob.append(prob.cpu())
            all_preds.append(pred.cpu())
            all_labels.append(label)

    all_prob = torch.cat(all_prob).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    metrics = evaluate_fn(model, test_loader, device, args.loss, num_classes)
    print(
        f"amae: {metrics['amae']}, wilson_idx: {metrics['wilson_idx']}, kendall's tau: {metrics['kendall_tau']}, balanced_accuracy: {metrics['balanced_accuracy']}"
    )

    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(conf_matrix)

    result = classification_report(all_labels, all_preds, digits=7)
    print(result)
