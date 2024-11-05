import torch
import torchvision
import torchvision.transforms as transforms


def auto_select_devices(
    num_clients: int, no_cuda: bool = False, no_mps: bool = False
) -> tuple[torch.device, list[torch.device]]:
    # The search order is cuda -> mps -> cpu.
    # If mutliple GPUs are available, we will rotate the devices.
    if not no_cuda and torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        return torch.device("cuda:0"), [
            torch.device(f"cuda:{i +1 % num_devices}") for i in range(num_clients)
        ]

    if not no_mps and torch.backends.mps.is_available():
        return torch.device("mps"), [torch.device("mps") for i in range(num_clients)]

    return torch.device("cpu"), [torch.device("cpu") for i in range(num_clients)]


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
