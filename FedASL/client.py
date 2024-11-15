import abc
from typing import Any, Iterator, Callable

import torch
from torch.utils.data import DataLoader

import FedASL.util as util


class FedClientBase:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        criterion: Any,
        accuracy_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device | str = "cpu",
    ):
        self.model = model
        self.dataloader = dataloader

        self._device = device

        self.criterion = criterion
        self.accuracy_func = accuracy_func

        self.data_iterator = self._get_train_batch_iterator()
        self.dtype = next(model.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    def _get_train_batch_iterator(self) -> Iterator:
        # NOTE: used only in init, will generate an infinite iterator from dataloader
        while True:
            for v in self.dataloader:
                yield v

    def get_next_input_labels(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch_inputs, labels = next(self.data_iterator)
        if self.device != torch.device("cpu") or self.dtype != torch.float32:
            batch_inputs = batch_inputs.to(self.device, self.dtype)
            labels = labels.to(self.device)
        return batch_inputs, labels

    @abc.abstractmethod
    def local_update(self, lr: float, local_update_steps: int) -> tuple[float, float]:
        """Local update steps at the client side"""

    def pull_model(self, server_model: torch.nn.Module) -> None:
        with torch.no_grad():
            for p, updated_p in zip(self.model.parameters(), server_model.parameters()):
                p.set_(updated_p.to(self._device))

    @abc.abstractmethod
    def push_step(self) -> torch.Tensor:
        """The push step after the local update is done."""


class FedASLClient(FedClientBase):
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        criterion: Any,
        accuracy_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device | str = "cpu",
    ):
        self.y = 0
        self.grad_prev = 0
        self.grad_curr = 0

        super().__init__(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            accuracy_func=accuracy_func,
            device=device,
        )

    def local_update(self, lr: float, local_update_steps: int) -> tuple[float, float]:
        train_loss = util.Metric("Client train loss")
        train_accuracy = util.Metric("Client train accuracy")
        self.y = 0
        for k in range(local_update_steps):
            if k > 0:
                # self.y is 0 when k==0. This can save the computation.
                util.set_flatten_model_back(
                    self.model,
                    util.get_flatten_model_param(self.model) - lr * self.y,
                )
            util.set_all_grad_zero(self.model)
            batch_inputs, labels = self.get_next_input_labels()
            pred = self.model(batch_inputs)
            loss = self.criterion(pred, labels)
            loss.backward()

            self.grad_curr = util.get_flatten_model_grad(self.model)
            self.y = self.y + self.grad_curr - self.grad_prev
            self.grad_prev = self.grad_curr

        train_loss.update(loss.detach().item())
        train_accuracy.update(self.accuracy_func(pred, labels).detach().item())

        return train_loss.avg, train_accuracy.avg

    def push_step(self) -> torch.Tensor:
        return self.y


class FedAvgClient(FedClientBase):
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        criterion: Any,
        accuracy_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device | str = "cpu",
    ):
        super().__init__(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            accuracy_func=accuracy_func,
            device=device,
        )

    def local_update(self, lr: float, local_update_steps: int) -> tuple[float, float]:
        train_loss = util.Metric("Client train loss")
        train_accuracy = util.Metric("Client train accuracy")
        self.y = 0
        for k in range(local_update_steps):
            util.set_all_grad_zero(self.model)
            batch_inputs, labels = self.get_next_input_labels()
            pred = self.model(batch_inputs)
            loss = self.criterion(pred, labels)
            loss.backward()

            with torch.no_grad():  # manually update model
                for p in self.model.parameters():
                    if p.requires_grad and p.grad is not None:
                        p.data.add_(p.grad, alpha=-lr)

        train_loss.update(loss.detach().item())
        train_accuracy.update(self.accuracy_func(pred, labels).detach().item())

        return train_loss.avg, train_accuracy.avg

    def push_step(self) -> torch.Tensor:
        return util.get_flatten_model_param(self.model)


class FedGaussianProjClient(FedClientBase):
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        criterion: Any,
        accuracy_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device | str = "cpu",
    ):
        super().__init__(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            accuracy_func=accuracy_func,
            device=device,
        )

    def local_update(
        self, lr: float, local_update_steps: int, seeds: list[int]
    ) -> tuple[float, float]:
        train_loss = util.Metric("Client train loss")
        train_accuracy = util.Metric("Client train accuracy")
        self.y = 0
        for k in range(local_update_steps):
            util.set_all_grad_zero(self.model)
            batch_inputs, labels = self.get_next_input_labels()
            pred = self.model(batch_inputs)
            loss = self.criterion(pred, labels)
            loss.backward()

            with torch.no_grad():
                grad_curr = util.get_flatten_model_grad(self.model)
                grad_proj = 0
                for seed in seeds:
                    torch.manual_seed(seed)
                    z = torch.randn_like(grad_curr)
                    proj_g = grad_curr.inner(z)
                    grad_proj += z.mul_(proj_g)
                grad_proj.div_(len(seeds))

                # manually update model
                util.set_flatten_model_back(
                    self.model, util.get_flatten_model_param(self.model) - lr * grad_proj
                )

        train_loss.update(loss.detach().item())
        train_accuracy.update(self.accuracy_func(pred, labels).detach().item())

        return train_loss.avg, train_accuracy.avg

    def push_step(self) -> torch.Tensor:
        return util.get_flatten_model_param(self.model)


class FedZOClient(FedClientBase):
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        criterion: Any,
        accuracy_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device | str = "cpu",
        mu: float = 1e-4,
    ):
        self.mu = mu
        super().__init__(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            accuracy_func=accuracy_func,
            device=device,
        )

    def perturb_model(self, flatten_weight: torch.Tensor, alpha: float, seed: int):
        torch.manual_seed(seed)
        z = torch.randn_like(flatten_weight)
        util.set_flatten_model_back(self.model, flatten_weight.add_(z, alpha=alpha))

    def local_update(
        self, lr: float, local_update_steps: int, seeds: list[int]
    ) -> tuple[float, float]:
        train_loss = util.Metric("Client train loss")
        train_accuracy = util.Metric("Client train accuracy")
        self.y = 0
        for k in range(local_update_steps):
            with torch.no_grad():
                batch_inputs, labels = self.get_next_input_labels()
                flatten_weight = util.get_flatten_model_param(self.model)
                update_delta = 0
                for seed in seeds:
                    self.perturb_model(flatten_weight, self.mu, seed=seed)
                    pred = self.model(batch_inputs)
                    loss_plus = self.criterion(pred, labels)
                    self.perturb_model(flatten_weight, -2 * self.mu, seed=seed)
                    pred = self.model(batch_inputs)
                    loss_minus = self.criterion(pred, labels)
                    self.perturb_model(flatten_weight, self.mu, seed=seed)
                    grad_scalar = (loss_plus - loss_minus) / (2 * self.mu)
                    torch.manual_seed(seed)
                    z = torch.randn_like(flatten_weight)
                    update_delta += grad_scalar * z
                # manually update model
                util.set_flatten_model_back(
                    self.model, flatten_weight - lr / len(seeds) * update_delta
                )

        pred = self.model(batch_inputs)
        loss = self.criterion(pred, labels)
        train_loss.update(loss.detach().item())
        train_accuracy.update(self.accuracy_func(pred, labels).detach().item())

        return train_loss.avg, train_accuracy.avg

    def push_step(self) -> torch.Tensor:
        return util.get_flatten_model_param(self.model)
