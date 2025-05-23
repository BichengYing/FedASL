import torch
import torchvision
import torchvision.transforms as transforms
import contextlib
import math
import numpy as np
from collections import defaultdict
from torch.utils.data.dataset import Subset


def auto_select_devices(
    num_clients: int, no_cuda: bool = False, no_mps: bool = False
) -> tuple[torch.device, list[torch.device]]:
    # The search order is cuda -> mps -> cpu.
    # If mutliple GPUs are available, we will rotate the devices.
    if not no_cuda and torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        return torch.device("cuda:0"), [
            torch.device(f"cuda:{(i+1) % num_devices}") for i in range(num_clients)
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
    alpha: float = None,
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

    if alpha is None:  # IID
        splitted_train_sets = torch.utils.data.random_split(
            train_dataset,
            [1 / num_clients for _ in range(num_clients)],  # Uniform split
            generator=torch.Generator().manual_seed(seed),
        )
    else:  # Non-IID
        labels = list(map(lambda x: x[1], train_dataset))
        splitted_train_sets = dirichlet_split(
            train_dataset,
            labels,
            num_clients,
            alpha,
            seed,
        )
    splitted_train_loaders = []
    for i in range(num_clients):
        dataloader = torch.utils.data.DataLoader(
            splitted_train_sets[i], batch_size=train_batch_size, **kwargs
        )
        splitted_train_loaders.append(dataloader)

    return splitted_train_loaders, test_loader


@contextlib.contextmanager
def temp_np_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def dirichlet_split(dataset, labels, num_split, alpha, random_seed):
    if num_split == 1:
        return [dataset]
    with temp_np_seed(random_seed):
        split_indexes = get_dirichlet_split_indexes(labels, num_split, alpha)
    return [Subset(dataset, split_indexes[i]) for i in range(num_split)]


def get_dirichlet_split_indexes(
    labels: list[int], num_split: int, alpha: float, balance_approach: bool = True
) -> list[list[int]]:
    """Given a list of labels and split it num_splits that follows Dirichlet(alpha*p) distribution.
    where p is the original class distribution in labels.

    Args:
        labels: A list of lables.
        num_split: Number of the split.
        alpha: A float number to indicate how non-iid the split it is. When alpha -> infinite, the
            returned index is more homogeneous. When alpha -> 0, the returned index is more hetero.
        balance_approach: If true, make sure all the clients has the same number of data. It will
            move the samples in the ranks having more randomly to the ranks having less.

    Returns:
        A list containing `num_split` sub-list.
    """
    n_data = len(labels)
    if num_split == 1:
        return [list(range(n_data))]

    label_indices_dict = defaultdict(list)
    for i, label in enumerate(labels):
        label_indices_dict[label].append(i)

    unique_labels = list(label_indices_dict.keys())
    n_labels = len(unique_labels)
    if not sorted(unique_labels) == list(range(n_labels)):
        raise ValueError("Please re-map the labels of dataset into " "0 to num_classes-1 integers.")

    data_class_prob = np.array([len(label_indices_dict[i]) for i in range(n_labels)]) / n_data
    # This is a matrix with dimension "num_split * num_classes"
    full_class_prob = np.random.dirichlet(alpha * data_class_prob, num_split)
    split_label_idx: list[list[int]] = [[] for _ in range(num_split)]

    for label, label_indices in label_indices_dict.items():
        cum_prob = np.cumsum(full_class_prob[:, label])
        normalized_cum_prob = (cum_prob / cum_prob[-1]).tolist()
        num_label_indices = len(label_indices)
        for rank in range(num_split):
            start_prob = 0 if rank == 0 else normalized_cum_prob[rank - 1]
            end_prob = normalized_cum_prob[rank]
            my_index_start = math.floor(start_prob * num_label_indices)
            my_index_end = math.floor(end_prob * num_label_indices)
            split_label_idx[rank].extend(label_indices[my_index_start:my_index_end])

    if balance_approach:
        expected_length = len(labels) // num_split
        diff_per_rank = [len(v) - expected_length for v in split_label_idx]
        extra_labels = []  # A temp buffer to store the extra labels moving between ranks.
        shuffle_extra_labels_once = True
        for rank in np.argsort(diff_per_rank)[::-1]:  # start from the client having most labels
            if diff_per_rank[rank] > 0:
                np.random.shuffle(split_label_idx[rank])
                extra_labels.extend(split_label_idx[rank][-diff_per_rank[rank] :])
                split_label_idx[rank] = split_label_idx[rank][: -diff_per_rank[rank]]
            elif diff_per_rank[rank] < 0:
                if shuffle_extra_labels_once:
                    np.random.shuffle(extra_labels)
                    shuffle_extra_labels_once = False
                split_label_idx[rank].extend(extra_labels[: abs(diff_per_rank[rank])])
                extra_labels = extra_labels[abs(diff_per_rank[rank]) :]

    return [sorted(v) for v in split_label_idx]
