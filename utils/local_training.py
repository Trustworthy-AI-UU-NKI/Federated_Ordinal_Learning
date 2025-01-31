import logging
import numpy as np
import copy

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer

from utils.ALA import ALA
from utils.utils import get_criterion


class LocalUpdate(object):
    def __init__(
        self, args, id, dataset, num_classes, device
    ):  # dataset now is already the local dataset
        self.args = args
        self.num_classes = num_classes
        self.id = id
        self.device = device
        self.local_dataset = dataset
        # self.class_num_list = self.get_num_class_list()
        self.class_num_list = self.local_dataset.get_num_class_list()
        logging.info(
            f"Client{id} ===> Each class num: {self.class_num_list}, Total: {len(self.local_dataset)}"
        )
        self.ldr_train = DataLoader(
            self.local_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )
        self.epoch = 0
        self.iter_num = 0
        self.lr = self.args.base_lr
        self.loss_name = args.loss
        self.loss = get_criterion(args.loss, self.num_classes, self.device)

    def train_fedprox(self, net, writer=None):
        net.train()
        self.loss.to(self.device)
        global_params = [val.detach().clone() for val in net.parameters()]
        # set the optimizer
        if self.args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4
            )
        elif self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                net.parameters(), lr=self.lr, weight_decay=1e-4
            )
        else:
            raise NotImplementedError
        print(f"Id: {self.id}, Num: {len(self.local_dataset)}")
        logging.info(f"Id: {self.id}, Num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for images, labels in self.ldr_train:
                if isinstance(images, list):
                    images = images[0]
                images, labels = images.to(self.device), labels.to(self.device)

                _, logits = net(images, project=False)
                if (
                    self.loss.__class__.__name__ == "OrdinalEncodingLoss"
                ):  # Adjust for your loss function name
                    loss, _ = self.loss(logits, labels)
                else:
                    loss = self.loss(logits, labels)

                self.optimizer.zero_grad()
                proximal_term = 0.0
                for local_weights, global_weights in zip(
                    net.parameters(), global_params
                ):
                    proximal_term += torch.square(
                        (local_weights - global_weights).norm(2)
                    )
                loss = loss + (self.args.mu / 2) * proximal_term
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                # logging.info(f"client{self.id}/loss_train", loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())
            logging.info(
                f"Epoch {epoch}, client{self.id}/loss: {np.array(batch_loss).mean()}"
            )

        return net.state_dict(), np.array(epoch_loss).mean()

    def train_avg(self, net, writer=None):
        net.train()
        self.loss.to(self.device)
        # set the optimizer
        if self.args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4
            )
        elif self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                net.parameters(), lr=self.lr, weight_decay=1e-4
            )
        else:
            raise NotImplementedError
        print(f"Id: {self.id}, Num: {len(self.local_dataset)}")
        logging.info(f"Id: {self.id}, Num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for images, labels in self.ldr_train:
                if isinstance(images, list):  # should be necessary, but let's see
                    images = images[0]
                images, labels = images.to(self.device), labels.to(self.device)

                _, logits = net(images, project=False)
                if (
                    self.loss.__class__.__name__ == "OrdinalEncodingLoss"
                ):  # Adjust for your loss function name
                    loss, _ = self.loss(logits, labels)
                else:
                    loss = self.loss(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                # logging.info(f"client{self.id}/loss_train", loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())
            logging.info(
                f"Epoch {epoch}, client{self.id}/loss: {np.array(batch_loss).mean()}"
            )

        return net.state_dict(), np.array(epoch_loss).mean()


class LocalUpdateMOON(object):
    def __init__(self, args, id, dataset, num_classes, device, net):
        self.args = args
        self.num_classes = num_classes
        self.id = id
        self.device = device
        self.local_dataset = dataset
        # self.class_num_list = self.get_num_class_list()
        self.class_num_list = self.local_dataset.get_num_class_list()
        self.old_model = copy.deepcopy(net)
        logging.info(
            f"Client{id} ===> Each class num: {self.class_num_list}, Total: {len(self.local_dataset)}"
        )
        self.ldr_train = DataLoader(
            self.local_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )
        self.epoch = 0
        self.iter_num = 0
        self.lr = self.args.base_lr
        self.loss_name = args.loss
        self.loss = get_criterion(args.loss, self.num_classes, self.device)

    def train_moon(self, net, writer=None):
        global_model = copy.deepcopy(net)
        self.old_model.eval()
        for param in self.old_model.parameters():
            param.requires_grad = False
        self.old_model.to(self.device)
        net.train()
        self.loss.to(self.device)
        # set the optimizer
        if self.args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, net.parameters()),
                lr=self.lr,
                betas=(0.9, 0.999),
                weight_decay=5e-4,
            )
        elif self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, net.parameters()),
                lr=self.lr,
                weight_decay=1e-4,
            )
        else:
            raise NotImplementedError
        print(f"Id: {self.id}, Num: {len(self.local_dataset)}")
        logging.info(f"Id: {self.id}, Num: {len(self.local_dataset)}")
        # train and update
        epoch_loss_total, epoch_loss_ce, epoch_loss_con = [], [], []
        for epoch in range(self.args.local_ep):
            batch_loss_total, batch_loss_ce, batch_loss_con = [], [], []
            for images, labels in self.ldr_train:
                if isinstance(images, list):
                    images = images[0]
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                rep, _ = net(images, project=True)  # check that this is correct
                _, logits = net(images, project=False)
                if (
                    self.loss.__class__.__name__ == "OrdinalEncodingLoss"
                ):  # Adjust for your loss function name
                    loss, _ = self.loss(logits, labels)
                else:
                    loss = self.loss(logits, labels)

                rep_old, _ = self.old_model(images, project=True)
                rep_old = rep_old.detach()
                rep_global, _ = global_model(images, project=True)
                rep_global = rep_global.detach()
                loss_con = -torch.log(
                    torch.exp(F.cosine_similarity(rep, rep_global) / self.args.tau)
                    / (
                        torch.exp(F.cosine_similarity(rep, rep_global) / self.args.tau)
                        + torch.exp(F.cosine_similarity(rep, rep_old) / self.args.tau)
                    )
                )
                loss_total = loss + self.args.mu * torch.mean(loss_con)

                # self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

                batch_loss_ce.append(loss.item())
                batch_loss_con.append(torch.mean(loss_con).item())
                batch_loss_total.append(loss_total.item())

                self.iter_num += 1
            self.epoch += 1
            epoch_loss_total.append(np.array(batch_loss_total).mean())
            epoch_loss_ce.append(np.array(batch_loss_ce).mean())
            epoch_loss_con.append(np.array(batch_loss_con).mean())
            logging.info(
                f"Epoch {epoch}, client{self.id}/total loss: {epoch_loss_total}, ce loss: {epoch_loss_ce}, con loss: {epoch_loss_con}"
            )

        self.old_model = copy.deepcopy(net)

        return net.state_dict(), np.array(epoch_loss_total).mean()

    # def train_moon(self, net, writer=None):
    #     global_model = copy.deepcopy(net)
    #     self.old_model.to(self.device)
    #     net.train()
    #     # set the optimizer
    #     if self.args.optimizer == "adam":
    #         self.optimizer = torch.optim.Adam(
    #             net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4
    #         )
    #     elif self.args.optimizer == "sgd":
    #         self.optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, weight_decay=1e-4)
    #     else:
    #         raise NotImplementedError
    #     print(f"Id: {self.id}, Num: {len(self.local_dataset)}")
    #     logging.info(f"Id: {self.id}, Num: {len(self.local_dataset)}")
    #     # train and update
    #     epoch_loss_total, epoch_loss_ce, epoch_loss_con = [], [], []
    #     ce_criterion = nn.CrossEntropyLoss()
    #     cos = nn.CosineSimilarity(dim=-1)
    #     for epoch in range(self.args.local_ep):
    #         batch_loss_total, batch_loss_ce, batch_loss_con = [], [], []
    #         for images, labels in self.ldr_train:
    #             if isinstance(images, list):
    #                 images = images[0]
    #             images, labels = images.to(self.device), labels.to(self.device)
    #             rep, logits = net(images, project=True)  # check that this is correct
    #             ce_loss = ce_criterion(logits, labels)

    #             rep_old, _ = self.old_model(images, project=True).detach()
    #             rep_global, _ = global_model(images, project=True).detach()
    #             posi = cos(rep, rep_global)
    #             logi = posi.reshape(-1, 1)
    #             nega = cos(rep, rep_old)
    #             logi = torch.cat((logi, nega.reshape(-1, 1)), dim=1)
    #             logi /= self.args.tau
    #             labels_con = torch.zeros(images.size(0)).cuda().long()
    #             loss_con = self.args.mu * ce_criterion(logi, labels_con)

    #             loss_total = ce_loss + loss_con

    #             self.optimizer.zero_grad()
    #             loss_total.backward()
    #             self.optimizer.step()

    #             batch_loss_ce.append(ce_loss.item())
    #             batch_loss_con.append(loss_con.item())
    #             batch_loss_total.append(loss_total.item())

    #             self.iter_num += 1
    #         self.epoch += 1
    #         epoch_loss_total.append(np.array(batch_loss_total).mean())
    #         epoch_loss_ce.append(np.array(batch_loss_ce).mean())
    #         epoch_loss_con.append(np.array(batch_loss_con).mean())
    #         logging.info(
    #             f"Epoch {epoch}, client{self.id}/total loss: {epoch_loss_total}, ce loss: {epoch_loss_ce}, con loss: {epoch_loss_con}"
    #         )

    #     self.old_model = copy.deepcopy(net)

    #     return net.state_dict(), np.array(epoch_loss_total).mean()


class LocalUpdateFedALA(object):
    def __init__(self, args, id, dataset, num_classes, device, net):
        self.args = args
        self.num_classes = num_classes
        self.id = id
        self.device = device
        self.local_dataset = dataset
        # self.class_num_list = self.get_num_class_list()
        self.class_num_list = self.local_dataset.get_num_class_list()
        self.old_model = copy.deepcopy(net)
        logging.info(
            f"Client{id} ===> Each class num: {self.class_num_list}, Total: {len(self.local_dataset)}"
        )
        self.ldr_train = DataLoader(
            self.local_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )
        self.epoch = 0
        self.iter_num = 0
        self.lr = self.args.base_lr
        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx
        self.loss_name = args.loss
        self.loss = get_criterion(args.loss, self.num_classes, self.device)
        self.ALA = ALA(
            self.id,
            self.loss,
            self.local_dataset,
            self.args.batch_size,
            self.rand_percent,
            self.layer_idx,
            self.eta,
            self.device,
        )

    def train_fedala(self, net, writer=None):
        net.train()
        self.loss.to(self.device)
        # set the optimizer
        if self.args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4
            )
        elif self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                net.parameters(), lr=self.lr, weight_decay=1e-4
            )
        else:
            raise NotImplementedError
        print(f"Id: {self.id}, Num: {len(self.local_dataset)}")
        logging.info(f"Id: {self.id}, Num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for images, labels in self.ldr_train:
                if isinstance(images, list):  # should be necessary, but let's see
                    images = images[0]
                images, labels = images.to(self.device), labels.to(self.device)

                _, logits = net(images)
                if (
                    self.loss.__class__.__name__ == "OrdinalEncodingLoss"
                ):  # Adjust for your loss function name
                    loss, _ = self.loss(logits, labels)
                else:
                    loss = self.loss(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                # logging.info(f"client{self.id}/loss_train", loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())
            logging.info(
                f"Epoch {epoch}, client{self.id}/loss: {np.array(batch_loss).mean()}"
            )

        return net.state_dict(), np.array(epoch_loss).mean()

    def local_initialization(self, local_model, received_global_model):
        return self.ALA.adaptive_local_aggregation(received_global_model, local_model)
