# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import logging
import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader, Subset
from typing import List, Tuple

from tqdm import tqdm


class ALA:
    def __init__(
        self,
        cid: int,
        loss: nn.Module,
        train_data: List[Tuple],
        batch_size: int,
        rand_percent: int,
        layer_idx: int = 0,
        eta: float = 1.0,
        device: str = "cpu",
        threshold: float = 0.1,
        num_pre_loss: int = 10,
    ) -> None:
        """
        Initialize ALA module

        Args:
            cid: Client ID.
            loss: The loss function.
            train_data: The reference of the local training data.
            batch_size: Weight learning batch size.
            rand_percent: The percent of the local training data to sample.
            layer_idx: Control the weight range. By default, all the layers are selected. Default: 0
            eta: Weight learning rate. Default: 1.0
            device: Using cuda or cpu. Default: 'cpu'
            threshold: Train the weight until the standard deviation of the recorded losses is less than a given threshold. Default: 0.1
            num_pre_loss: The number of the recorded losses to be considered to calculate the standard deviation. Default: 10

        Returns:
            None.
        """

        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.weights = None  # Learnable local aggregation weights.
        self.start_phase = True

    def adaptive_local_aggregation(self, global_model: nn.Module, local_model: nn.Module) -> None:
        """
        Generates the Dataloader for the randomly sampled local training data and
        preserves the lower layers of the update.

        Args:
            global_model: The received global/aggregated model.
            local_model: The trained local model.

        Returns:
            None.
        """

        # randomly sample partial local training data
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio * len(self.train_data))
        rand_idx = random.randint(0, len(self.train_data) - rand_num)
        # rand_loader = DataLoader(self.train_data[rand_idx:rand_idx+rand_num], self.batch_size, drop_last=False)
        # my data is not a numpy array like in the original library, it's a pytorch dataset insteas, so we cannot slice it
        # we need to create a subset
        rand_indices = torch.arange(rand_idx, rand_idx + rand_num).tolist()
        subset = Subset(self.train_data, rand_indices)
        rand_loader = DataLoader(subset, batch_size=self.batch_size, drop_last=False)

        # obtain the references of the parameters
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # deactivate ALA at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return local_model.state_dict()

        # preserve all the updates in the lower layers
        for param, param_g in zip(params[: -self.layer_idx], params_g[: -self.layer_idx]):
            param.data = param_g.data.clone()

        # temp local model only for weight learning
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())
        selected_params = params_t[-self.layer_idx :]
        for param in selected_params:
            print(param.shape)

        # only consider higher layers
        params_p = params[-self.layer_idx :]
        params_gp = params_g[-self.layer_idx :]
        params_tp = params_t[-self.layer_idx :]

        # frozen the lower layers to reduce computational cost in Pytorch
        for param in params_t[: -self.layer_idx]:
            param.requires_grad = False

        # for param in params_tp:
        #     param.requires_grad = True

        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # initialize the weight to all ones in the beginning
        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
            param_t.data = param + (param_g - param) * weight

        # weight learning
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        while True:
            for x, y in rand_loader:
                if isinstance(x, list):
                    x = x[0]
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                _, output = model_t(x)
                if (
                    self.loss.__class__.__name__ == "OrdinalEncodingLoss"
                ):  # Adjust for your loss function name
                    loss_value, _ = self.loss(output, y)
                else:
                    loss_value = self.loss(output, y)

                loss_value.backward()

                # update weight in this batch
                # for param_t in params_tp:
                #     if param_t.grad is None:
                #         raise ValueError("Gradient is None for param_t")

                for param_t, param, param_g, weight in zip(
                    params_tp, params_p, params_gp, self.weights
                ):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1
                    )

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(
                    params_tp, params_p, params_gp, self.weights
                ):
                    param_t.data = param + (param_g - param) * weight

            losses.append(loss_value.item())
            cnt += 1

            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                break

            # train the weight until convergence
            if (
                len(losses) > self.num_pre_loss
                and np.std(losses[-self.num_pre_loss :]) < self.threshold
            ):
                logging.info(
                    f"Client: {self.cid}, \tStd: {np.std(losses[-self.num_pre_loss :])}, \tALA epochs: {cnt}"
                )
                break

        self.start_phase = False

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()

        return local_model.state_dict()
