from __future__ import annotations
import abc
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

from FedASL import util
from FedASL import client as fl_client


class FedServerBase:
    def __init__(
        self,
        clients: Sequence[fl_client.FedClientBase],
        device: torch.device,
        server_model: torch.nn.Module,
        server_criterion: Any,
        server_accuracy_func: Callable,
        num_sample_clients: int = 10,
        local_update_steps: int = 10,
    ) -> None:
        self.clients = clients
        self.device = device
        self.num_sample_clients = num_sample_clients
        self.local_update_steps = local_update_steps

        self.server_model = server_model
        self.server_criterion = server_criterion
        self.server_accuracy_func = server_accuracy_func

        self.dtype = next(server_model.parameters()).dtype

    def get_sampled_client_index(self, prob: Sequence[float], participation: str) -> list[int]:
        assert len(self.clients) == len(prob)
        # TODO(ybc) make this function control from the outside
        clients_list = list(self.clients)
        if participation == "bern":
            list_index = np.where(prob == 1)[0]
        elif participation == "markov":
            list_index = [i for i, value in enumerate(prob) if value == 1]
        elif participation == "uniform":
            list_index = np.random.choice(
                np.arange(len(self.clients)), replace=True, size=self.num_sample_clients, p=prob
            )
        selected_clients = {clients_list[i] for i in list_index}
        return selected_clients

    @abc.abstractmethod
    def train_one_step(
        self, sampling_prob: Sequence[float], participation: str
    ) -> tuple[float, float]:
        """One step for the training"""

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


class FedASLServer(FedServerBase):
    def __init__(
        self,
        clients: Sequence[fl_client.FedASLClient],
        device: torch.device,
        server_model: torch.nn.Module,
        server_criterion: Any,
        server_accuracy_func: Callable,
        num_sample_clients: int = 10,
        local_update_steps: int = 10,
    ) -> None:
        self.y = 0
        super().__init__(
            clients=clients,
            device=device,
            server_model=server_model,
            server_criterion=server_criterion,
            server_accuracy_func=server_accuracy_func,
            num_sample_clients=num_sample_clients,
            local_update_steps=local_update_steps,
        )

    def train_one_step(
        self, lr: float, sampling_prob: Sequence[float], participation: str
    ) -> tuple[float, float]:
        sampled_clients: list[int] = self.get_sampled_client_index(sampling_prob, participation)
        if not sampled_clients:
            return np.nan, np.nan

        step_train_loss = util.Metric("train_loss")
        step_train_accuracy = util.Metric("train_loss")
        for client in sampled_clients:
            client.pull_model(self.server_model)
            client_loss, client_accuracy = client.local_update(lr, self.local_update_steps)
            self.y += client.push_step().to(self.device)

            step_train_loss.update(client_loss)
            step_train_accuracy.update(client_accuracy)

        # Update the server model
        util.set_flatten_model_back(
            self.server_model, util.get_flatten_model_param(self.server_model) - lr * self.y
        )

        return step_train_loss.avg, step_train_accuracy.avg


class FedAvgServer(FedServerBase):
    def __init__(
        self,
        clients: Sequence[fl_client.FedAvgClient],
        device: torch.device,
        server_model: torch.nn.Module,
        server_criterion: Any,
        server_accuracy_func: Callable,
        num_sample_clients: int = 10,
        local_update_steps: int = 10,
    ) -> None:
        super().__init__(
            clients=clients,
            device=device,
            server_model=server_model,
            server_criterion=server_criterion,
            server_accuracy_func=server_accuracy_func,
            num_sample_clients=num_sample_clients,
            local_update_steps=local_update_steps,
        )

    def train_one_step(
        self, lr: float, sampling_prob: Sequence[float], participation: str
    ) -> tuple[float, float]:
        sampled_clients: list[int] = self.get_sampled_client_index(sampling_prob, participation)
        if not sampled_clients:
            return np.nan, np.nan

        step_train_loss = util.Metric("train_loss")
        step_train_accuracy = util.Metric("train_loss")
        for client in sampled_clients:
            client.pull_model(self.server_model)
            client_loss, client_accuracy = client.local_update(lr, self.local_update_steps)

            step_train_loss.update(client_loss)
            step_train_accuracy.update(client_accuracy)

        # Update the server model
        avg_model = 0
        for client in sampled_clients:
            avg_model += client.push_step()
        avg_model.div_(len(sampled_clients))
        util.set_flatten_model_back(self.server_model, avg_model)

        return step_train_loss.avg, step_train_accuracy.avg


class FedAUServer(FedServerBase):
    def __init__(
        self,
        clients: Sequence[fl_client.FedAUClient],
        device: torch.device,
        server_model: torch.nn.Module,
        server_criterion: Any,
        server_accuracy_func: Callable,
        num_sample_clients: int = 10,
        local_update_steps: int = 10,
    ) -> None:
        super().__init__(
            clients=clients,
            device=device,
            server_model=server_model,
            server_criterion=server_criterion,
            server_accuracy_func=server_accuracy_func,
            num_sample_clients=num_sample_clients,
            local_update_steps=local_update_steps,
        )

    def train_one_step(
        self, lr: float, sampling_prob: Sequence[float], participation: str
    ) -> tuple[float, float]:
        sampled_clients: list[int] = self.get_sampled_client_index(sampling_prob, participation)
        if not sampled_clients:
            return np.nan, np.nan

        step_train_loss = util.Metric("train_loss")
        step_train_accuracy = util.Metric("train_loss")
        for client in sampled_clients:
            client.pull_model(self.server_model)
            client_loss, client_accuracy = client.local_update(lr, self.local_update_steps)
            step_train_loss.update(client_loss)
            step_train_accuracy.update(client_accuracy)

        for client in self.clients:
            if client in sampled_clients:
                client.update_weight(participated=1)
            else:
                client.update_weight(participated=0)

        # Update the server model
        weighted_avg_model = 0
        total_weight = 0
        for client in sampled_clients:
            model, weight = client.push_step()
            # print(weight)
            weighted_avg_model += model * weight
            total_weight += weight
        weighted_avg_model.div_(total_weight)
        util.set_flatten_model_back(self.server_model, weighted_avg_model)

        return step_train_loss.avg, step_train_accuracy.avg


# Theoratical we just need seed to communicate between server and client. Here
# we just use FedAvg for simplicity since it is equivalent.
class FedGaussianProjServer(FedServerBase):
    def __init__(
        self,
        clients: Sequence[fl_client.FedGaussianProjClient],
        device: torch.device,
        server_model: torch.nn.Module,
        server_criterion: Any,
        server_accuracy_func: Callable,
        num_sample_clients: int = 10,
        local_update_steps: int = 10,
        num_pert: int = 10,
    ) -> None:
        self.num_pert = num_pert
        super().__init__(
            clients=clients,
            device=device,
            server_model=server_model,
            server_criterion=server_criterion,
            server_accuracy_func=server_accuracy_func,
            num_sample_clients=num_sample_clients,
            local_update_steps=local_update_steps,
        )

    def train_one_step(self, lr: float, sampling_prob: Sequence[float]) -> tuple[float, float]:
        sampled_clients: list[int] = self.get_sampled_client_index(sampling_prob)

        step_train_loss = util.Metric("train_loss")
        step_train_accuracy = util.Metric("train_loss")
        for client in sampled_clients:
            seeds = np.random.randint(100000, size=self.num_pert)
            client.pull_model(self.server_model)
            client_loss, client_accuracy = client.local_update(lr, self.local_update_steps, seeds)

            step_train_loss.update(client_loss)
            step_train_accuracy.update(client_accuracy)

        # Update the server model
        avg_model = 0
        for client in sampled_clients:
            avg_model += client.push_step()
        avg_model.div_(len(sampled_clients))
        util.set_flatten_model_back(self.server_model, avg_model)

        return step_train_loss.avg, step_train_accuracy.avg


class FedZOServer(FedServerBase):
    def __init__(
        self,
        clients: Sequence[fl_client.FedZOClient],
        device: torch.device,
        server_model: torch.nn.Module,
        server_criterion: Any,
        server_accuracy_func: Callable,
        num_sample_clients: int = 10,
        local_update_steps: int = 10,
        num_pert: int = 10,
        same_seed: bool = True,
    ) -> None:
        self.num_pert = num_pert
        self.same_seed = same_seed
        super().__init__(
            clients=clients,
            device=device,
            server_model=server_model,
            server_criterion=server_criterion,
            server_accuracy_func=server_accuracy_func,
            num_sample_clients=num_sample_clients,
            local_update_steps=local_update_steps,
        )

    def train_one_step(self, lr: float, sampling_prob: Sequence[float]) -> tuple[float, float]:
        sampled_clients: list[int] = self.get_sampled_client_index(sampling_prob)

        step_train_loss = util.Metric("train_loss")
        step_train_accuracy = util.Metric("train_loss")
        for i, client in enumerate(sampled_clients):
            seeds = np.random.randint(100000, size=self.num_pert)
            if not self.same_seed:
                seeds += i
            client.pull_model(self.server_model)
            client_loss, client_accuracy = client.local_update(lr, self.local_update_steps, seeds)

            step_train_loss.update(client_loss)
            step_train_accuracy.update(client_accuracy)

        # Update the server model
        avg_model = 0
        for client in sampled_clients:
            avg_model += client.push_step()
        avg_model.div_(len(sampled_clients))
        util.set_flatten_model_back(self.server_model, avg_model)

        return step_train_loss.avg, step_train_accuracy.avg


class ScaffoldServer(FedServerBase):
    def __init__(
        self,
        clients: Sequence[fl_client.FedAvgClient],
        device: torch.device,
        server_model: torch.nn.Module,
        server_criterion: Any,
        server_accuracy_func: Callable,
        num_sample_clients: int = 10,
        local_update_steps: int = 10,
        global_step: float = 1.0,
    ) -> None:
        self.global_c = 0
        self.global_step = global_step
        super().__init__(
            clients=clients,
            device=device,
            server_model=server_model,
            server_criterion=server_criterion,
            server_accuracy_func=server_accuracy_func,
            num_sample_clients=num_sample_clients,
            local_update_steps=local_update_steps,
        )

    def train_one_step(
        self, lr: float, sampling_prob: Sequence[float], participation: str
    ) -> tuple[float, float]:
        sampled_clients: list[int] = self.get_sampled_client_index(sampling_prob, participation)
        if not sampled_clients:
            return np.nan, np.nan

        step_train_loss = util.Metric("train_loss")
        step_train_accuracy = util.Metric("train_loss")
        for client in sampled_clients:
            client.pull_model(self.server_model)
            client_loss, client_accuracy = client.local_update(
                lr, self.local_update_steps, self.global_c
            )

            step_train_loss.update(client_loss)
            step_train_accuracy.update(client_accuracy)

        server_param = util.get_flatten_model_param(self.server_model)
        # Update the server model
        for client in sampled_clients:
            client_model, delta_c = client.push_step()
            server_param.add_(client_model.mul_(self.global_step / self.num_sample_clients))
            self.global_c += delta_c.mul_(1 / len(self.clients))

        util.set_flatten_model_back(self.server_model, server_param)

        return step_train_loss.avg, step_train_accuracy.avg
