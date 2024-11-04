from __future__ import annotations

import random
from typing import Any, Callable, Iterable, Sequence

import torch

import FedASL.util as util
from FedASL.client import FedASLClient


class FedASLServer:
    def __init__(
        self,
        clients: Sequence[FedASLClient],
        device: torch.device,
        lr: float,
        server_model: torch.nn.Module,
        server_criterion: Any,
        server_accuracy_func: Callable,
        num_sample_clients: int = 10,
        local_update_steps: int = 10,
    ) -> None:
        self.clients = clients
        self.device = device
        self.lr = lr
        self.num_sample_clients = num_sample_clients
        self.local_update_steps = local_update_steps

        self.server_model = server_model
        self.server_criterion = server_criterion
        self.server_accuracy_func = server_accuracy_func

        self.dtype = next(server_model.parameters()).dtype

        self.y = 0

    def get_sampled_client_index(self) -> list[int]:
        # TODO(ybc) make this function control from the outside
        return random.sample(range(len(self.clients)), self.num_sample_clients)

    def set_learning_rate(self, lr: float) -> None:
        # Client
        for client in self.clients:
            for p in client.optimizer.param_groups:
                p["lr"] = lr
        # Server
        if self.server_model:
            for p in self.optim.param_groups:
                p["lr"] = lr

    def train_one_step(self) -> tuple[float, float]:
        sampled_client_indices: list[int] = self.get_sampled_client_index()

        step_train_loss = util.Metric("train_loss")
        step_train_accuracy = util.Metric("train_loss")
        for index in sampled_client_indices:
            client = self.clients[index]

            client.pull_model(self.server_model)
            client_loss, client_accuracy = client.local_update(self.local_update_steps)
            self.y += client.push_grad().to(self.device)

            step_train_loss.update(client_loss)
            step_train_accuracy.update(client_accuracy)

        # Update the server model
        util.set_flatten_model_back(
            self.server_model, util.get_flatten_model_param(self.server_model) - self.lr * self.y
        )

        return step_train_loss.avg, step_train_accuracy.avg

    def eval_model(self, test_loader: Iterable[Any], iteration: int) -> tuple[float, float]:
        self.server_model.eval()
        eval_loss = util.Metric("Eval loss")
        eval_accuracy = util.Metric("Eval accuracy")
        with torch.no_grad():
            for _, (batch_inputs, batch_labels) in enumerate(test_loader):
                if self.device != torch.device("cpu") or self.dtype != torch.float32:
                    batch_inputs = batch_inputs.to(self.device, self.dtype)
                    batch_labels = batch_labels.to(self.device)
                pred = self.server_model(batch_inputs)
                eval_loss.update(self.server_criterion(pred, batch_labels))
                eval_accuracy.update(self.server_accuracy_func(pred, batch_labels))
        print(
            f"\nEvaluation(Iteration {iteration + 1}): ",
            f"Eval Loss:{eval_loss.avg:.4f}, " f"Accuracy:{eval_accuracy.avg * 100:.2f}%",
        )
        return eval_loss.avg, eval_accuracy.avg
