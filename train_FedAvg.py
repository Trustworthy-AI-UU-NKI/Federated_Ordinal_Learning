import os
import re
import sys
import copy
import logging
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import itertools as I

# from tensorboardX import SummaryWriter

from utils import get_imageids_and_labels_per_centre
from utils.FedAvg import FedAvg
from dataset.get_dataset import prepare_transforms
from dataset.dataset import CSAWM
from val import compute_loss_of_classes, evaluate_fn
from networks.networks import ResNetWithProjector
from utils.local_training import LocalUpdate
from utils.utils import set_seed, get_num_classes


# BIG TODO: now we make it work for ce and bce, but there are some parts where having we
# will porbably have go separate num_class in two variables num_class_dataset and num_class_loss
# to make it work with ordinal encoding


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CSAWM", help="dataset name")
    parser.add_argument("--exp", type=str, default="FedIIC", help="experiment name")
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size per gpu")
    parser.add_argument(
        "--base_lr", type=float, default=1e-5, help="base learning rate"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="parameter for non-iid"
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=2.0,
        help="weight for Intra-client contrastive learning",
    )
    parser.add_argument(
        "--k2",
        type=float,
        default=2.0,
        help="weight for Inter-client contrastive learning",
    )
    parser.add_argument("--d", type=float, default=0.25, help="difficulty")
    parser.add_argument(
        "--deterministic",
        type=int,
        default=1,
        help="whether use deterministic training",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gpu", type=str, default="1", help="GPU to use")
    parser.add_argument("--local_ep", type=int, default=1, help="local epoch")
    parser.add_argument("--rounds", type=int, default=200, help="rounds")
    # Added by me
    parser.add_argument(
        "--num-folds",
        dest="num_folds",
        type=int,
        default=1,
        help="Number of folds to run",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        help="Loss to use: ce, ordinal_encoding, binomial_unimodal",
    )
    parser.add_argument(
        "--path-to-splits",
        dest="path_to_splits",
        type=str,
        help="Path to splits csv file",
        default="/projects/federated_ordinal_classification/FederatedOrdinality/data/splits/csaw_m/actual_splits",
    )
    parser.add_argument(
        "--cardinality",
        type=str,
        default="classification_8_classes",
        help="number of classes to consider",
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        help="Number of workers",
        default=16,
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        type=str,
        help="Path to the data",
        default="/processing/v.corbetta/CSAW-M/images/preprocessed/train",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="type of optimizer to use: adam or sgd",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = args_parser()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    # ------------------------------ deterministic or not ------------------------------
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args)

    # ------------------------------ output files ------------------------------
    ROOT_DIR = Path(__file__).parent.parent
    alpha_p_combinations = [
        [0.9, 0.8],
    ]  # , [1.5, 0.85], [2.0, 0.9], [5.0, 0.95], [10.0, 0.99]]
    folds = [1, 3]  # [0, 1, 3]
    if args.optimizer == "adam":
        lr_values = [1e-5]
    elif args.optimizer == "sgd":
        lr_values = [0.03, 1e-3, 1e-1]
    rounds_and_local_epochs = [(20, 5)]  # [(100, 1), (20, 5), (10, 10)]
    losses = ["ce", "binomial", "ordinal_encoding"]
    grid_search = I.product(lr_values, rounds_and_local_epochs, losses)
    classification_csv = pd.read_csv(
        Path(args.path_to_splits) / f"{args.cardinality}.csv"
    )
    num_classes = int(
        re.search(r"classification_(\d+)_classes", args.cardinality).group(1)
    )
    args.num_classes = num_classes
    for base_lr, round_and_local_epoch, loss in grid_search:
        args.base_lr = base_lr
        args.rounds, args.local_ep = round_and_local_epoch
        args.loss = loss
        hyperparameter_setup = (
            f"{args.optimizer}_{args.base_lr}_{args.local_ep}_{args.rounds}"
        )
        outputs_dir = ROOT_DIR / "results" / f"fedavg_{args.loss}_check_check"
        for alpha, p in alpha_p_combinations:
            print(f"\nProcessing experiments for Alpha: {alpha}, P: {p}")
            for fold in folds:
                exp_dir = (
                    outputs_dir
                    / hyperparameter_setup
                    / f"{args.dataset}_{alpha}_{p}_{args.loss}"
                    / f"{fold}"
                    / "train"
                )
                exp_dir.mkdir(exist_ok=True, parents=True)
                ckpt_dir = exp_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True, parents=True)

                logging.basicConfig(
                    filename=exp_dir / "logs.txt",
                    level=logging.INFO,
                    format="[%(asctime)s.%(msecs)03d] %(message)s",
                    datefmt="%H:%M:%S",
                )
                logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
                logging.info(str(args))
                writer = None  # SummaryWriter(tensorboard_dir)

                # ------------------------------ dataset and dataloader ------------------------------
                image_ids_labels_per_centre = get_imageids_and_labels_per_centre(
                    classification_csv, fold, alpha, p
                )
                _val_datasets = []
                _train_datasets = []
                # train_augs, val_augs = prepare_transforms()
                train_augs, val_augs = prepare_transforms()
                for centre, image_ids_labels in image_ids_labels_per_centre.items():
                    df = pd.DataFrame(image_ids_labels, columns=["Filename", "Label"])
                    train_df, val_df = train_test_split(
                        df, test_size=0.2, random_state=args.seed, stratify=df["Label"]
                    )
                    print(centre)
                    print(f"{train_df.head()=}")
                    print(f"{val_df.head()=}")
                    train_dataset = CSAWM(
                        train_df, args.data_dir, num_classes, train_augs, args.loss
                    )
                    _val_dataset = CSAWM(
                        val_df, args.data_dir, num_classes, val_augs, args.loss
                    )
                    _train_datasets.append(train_dataset)
                    _val_datasets.append(_val_dataset)

                val_dataset = ConcatDataset(_val_datasets)
                val_loader = DataLoader(
                    dataset=val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    drop_last=True,
                )

                if args.dataset == "isic2019":
                    args.n_clients = 10
                elif args.dataset == "ich":
                    args.n_clients = 20
                else:
                    args.n_clients = 5
                # What did I modify? We now have a list of datasets for the training of the clients, validation is like in their code basically

                # ------------------------------ global and local settings ------------------------------
                num_classes = get_num_classes(args.loss, args.num_classes)
                net_glob = ResNetWithProjector(args=args).to(
                    device
                )  # TODO: for now n_classes is only used to instantiate the model, still compatible with OE
                print(net_glob)
                net_glob.train()
                w_glob = net_glob.state_dict()
                w_locals = []
                trainer_locals = []
                net_locals = []

                dict_len = [len(dataset) for dataset in _train_datasets]

                for id in range(args.n_clients):
                    trainer_locals.append(
                        LocalUpdate(
                            args,
                            id,
                            copy.deepcopy(_train_datasets[id]),
                            num_classes,
                            device,
                        )
                    )  # Modified so that it already takes tha lists of data loaders for each client, while now the splits are performed inside LocalUpdate
                    w_locals.append(copy.deepcopy(w_glob))
                    net_locals.append(copy.deepcopy(net_glob).to(device))
                print(f"Done with instantiating clients")
                logging.info("Done with instantiating clients")

                # ------------------------------ begin training ------------------------------
                best_performance = (
                    1.0  # my optimization is on auoc which must be minimised
                )
                lr = args.base_lr
                auoc = []
                for com_round in range(args.rounds):
                    logging.info(
                        f"\n======================> round: {com_round} <======================"
                    )
                    loss_locals = []
                    # writer.add_scalar("train/lr", lr, com_round)

                    # with torch.no_grad():
                    #     class_embedding = w_glob["model.fc.weight"].detach().clone().to(device)
                    #     feature_avg = net_glob.projector(class_embedding).detach().clone()
                    # logging.info("similarity before")
                    # logging.info(
                    #     torch.matmul(F.normalize(feature_avg, dim=1), F.normalize(feature_avg, dim=1).T)
                    # )
                    # feature_avg.requires_grad = True
                    # optimizer_f = torch.optim.SGD([feature_avg], lr=0.1)
                    # mask = torch.ones((n_classes, n_classes)) - torch.eye(
                    #     (n_classes)
                    # )  # TODO: check if this is compatible with ordinal encoding
                    # mask = mask.to(
                    #     device
                    # )  # Mask is used to esclude self-similarities in the similarity computation
                    # for i in range(1000):
                    #     feature_avg_n = F.normalize(feature_avg, dim=1)
                    #     cos_sim = torch.matmul(feature_avg_n, feature_avg_n.T)
                    #     cos_sim = ((cos_sim * mask).max(1)[0]).sum()
                    #     optimizer_f.zero_grad()
                    #     cos_sim.backward()
                    #     optimizer_f.step()
                    # logging.info("similarity after")
                    # logging.info(
                    #     torch.matmul(F.normalize(feature_avg, dim=1), F.normalize(feature_avg, dim=1).T)
                    # )

                    # loss_matrix = torch.zeros(args.n_clients, n_classes)
                    # print("Created loss matrix")
                    # class_num = torch.zeros(args.n_clients, n_classes)
                    # print("Initialised class_num")
                    net_glob = net_glob.to(device)
                    print("Moved model to device")
                    # for id in range(args.n_clients):
                    #     class_num[id] = torch.tensor(trainer_locals[id].local_dataset.get_num_class_list())
                    #     dataset_client = TensorDataset(images_all[id], labels_all[id])
                    #     dataLoader_client = DataLoader(
                    #         dataset_client, batch_size=args.batch_size, shuffle=False
                    #     )
                    #     loss_matrix[id] = compute_loss_of_classes(
                    #         net_glob, dataLoader_client, n_classes, device
                    #     )
                    # for id in tqdm(range(args.n_clients)):
                    #     # class_num[id] = torch.tensor(
                    #     #     trainer_locals[id].local_dataset.get_num_class_list()
                    #     # )
                    #     dataLoader_client = DataLoader(
                    #         _train_datasets[id], batch_size=args.batch_size, shuffle=False
                    #     )
                    # loss_matrix[id] = compute_loss_of_classes(
                    #     net_glob, dataLoader_client, n_classes, device
                    # )
                    # print(f"Done computing loss per class: {loss_matrix[id]}")
                    # num = torch.sum(class_num, dim=0, keepdim=True)
                    # logging.info("class-num of this round")
                    # logging.info(num)
                    # loss_matrix = loss_matrix / (1e-5 + num)
                    # loss_class = torch.sum(
                    #     loss_matrix, dim=0
                    # )  # we are just normalising the loss based on the number of samples per class
                    # logging.info("loss of this round")
                    # logging.info(loss_class)  # loss_class is used in DALA

                    # local training
                    for id in range(args.n_clients):
                        trainer_locals[id].lr = lr
                        local = trainer_locals[id]
                        # local.loss_class = loss_class
                        net_local = net_locals[id]
                        w, loss = local.train_avg(copy.deepcopy(net_local))
                        w_locals[id] = copy.deepcopy(w)
                        loss_locals.append(copy.deepcopy(loss))

                    # upload and download
                    with torch.no_grad():
                        w_glob = FedAvg(w_locals, dict_len)
                    net_glob.load_state_dict(w_glob)
                    for id in range(args.n_clients):
                        net_locals[id].load_state_dict(w_glob)

                    # global validation
                    net_glob = net_glob.to(device)
                    # bacc_g, conf_matrix = compute_bacc(net_glob, val_loader, get_confusion_matrix=True, args=args)
                    metrics = evaluate_fn(
                        net_glob, val_loader, device, args.loss, num_classes
                    )
                    logging.info(
                        f"Round {com_round}: amae: {metrics['amae']}, wilson_idx: {metrics['wilson_idx']}, kendall's tau: {metrics['kendall_tau']}, balanced_accuracy: {metrics['balanced_accuracy']}"
                    )

                    # save model
                    if metrics["wilson_idx"] < best_performance:
                        best_performance = metrics["wilson_idx"]
                        torch.save(
                            net_glob.state_dict(),
                            ckpt_dir
                            / f"best_model_{com_round}_{best_performance}.ckpt",
                        )
                        torch.save(net_glob.state_dict(), ckpt_dir / "best_model.ckpt")
                    logging.info(
                        f"best bacc: {best_performance}, now bacc: {metrics['wilson_idx']}"
                    )
                    auoc.append(metrics["wilson_idx"])

                # writer.close()
                logging.info(auoc)
