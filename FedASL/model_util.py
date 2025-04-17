import torch
import copy
from FedASL import models
from FedASL.models.resnet import ResNet20, ResNet32, ResNet56, _weights_init
# from FedASL.models.resnet8 import ResNet8, _weights_init

import torchvision.models as modell
import torch.nn as nn


def prepare_models(
    dataset_name: str,
    num_clients: int,
    client_devices: list[torch.device],
    server_device: torch.device,
    model_dtype: str,
) -> tuple[list[torch.nn.Module], torch.nn.Module]:
    torch_dtype = {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[model_dtype]

    if dataset_name == "mnist":
        server_model = models.CNN_MNIST().to(torch_dtype).to(server_device)
        client_models = [models.CNN_MNIST().to(torch_dtype).to(device) for device in client_devices]
    elif dataset_name == "cifar10":
        server_model = ResNet32(num_classes=10).to(torch_dtype).to(server_device)
        server_model.apply(_weights_init)
        client_models = [
            ResNet32(num_classes=10).to(torch_dtype).to(device) for device in client_devices
        ]
        [model.apply(_weights_init) for model in client_models]
    elif dataset_name == "cifar100": 
        server_model = ResNet20(num_classes=100).to(torch_dtype).to(server_device)
        client_models = [
            ResNet20(num_classes=100).to(torch_dtype).to(device) for device in client_devices
        ]
    elif dataset_name == "fashion":
        server_model = models.CNN_FMNIST().to(torch_dtype).to(server_device)
        client_models = [
            models.CNN_FMNIST().to(torch_dtype).to(device) for device in client_devices
        ]
    return client_models, server_model
