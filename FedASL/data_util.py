import torch
import torchvision
import torchvision.transforms as transforms


def prepare_dataloaders(
    dataset_name: str,
    num_clients: int,
    train_batch_size: int,
    test_batch_size: int = 1000,
    seed: int = 12345,
    **kwargs,
) -> tuple[list[torch.utils.data.DataLoader], torch.utils.data.DataLoader]:
    if dataset_name == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, **kwargs
        )
    elif dataset_name == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, **kwargs
        )
    elif dataset_name == "fashion":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
        )
        train_dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, **kwargs
        )
    else:
        raise Exception(f"Dataset {dataset_name} is not supported")

    splitted_train_sets = torch.utils.data.random_split(
        train_dataset,
        [1 / num_clients for _ in range(num_clients)],  # uniform split
        generator=torch.Generator().manual_seed(seed),
    )
    splitted_train_loaders = []
    for i in range(num_clients):
        dataloader = torch.utils.data.DataLoader(
            splitted_train_sets[i], batch_size=train_batch_size, **kwargs
        )
        splitted_train_loaders.append(dataloader)

    return splitted_train_loaders, test_loader
