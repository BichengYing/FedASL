import argparse
import torch
from tqdm import tqdm

from FedASL import data_util
from FedASL import model_util
from FedASL import util
from FedASL.client import FedASLClient
from FedASL.server import FedASLServer


def setup_clients(
    num_clients: int,
    client_models: list[torch.nn.Module],
    train_loaders: list[torch.utils.data.DataLoader],
    lr: float,
) -> FedASLServer:
    assert len(train_loaders) == num_clients
    client_device = torch.device("mps")
    clients = []

    for i in range(args.num_clients):
        client = FedASLClient(
            client_models[i],
            train_loaders[i],
            lr=lr,
            criterion=torch.nn.CrossEntropyLoss(),
            accuracy_func=util.accuracy,
            device=client_device,
        )
        clients.append(client)

    return clients


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedASL trainning")
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--num_sample_clients", type=int, default=2)
    parser.add_argument("--local_update", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="mnist", help="[mnist, fashion, cifar10]")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--dtype", type=str, default="float32", help="random seed")
    parser.add_argument("--eval-iterations", type=int, default=25)

    args = parser.parse_args()

    client_devices = [torch.device("mps") for _ in range(args.num_clients)]
    server_device = torch.device("mps")
    train_loaders, test_loader = data_util.prepare_dataloaders(
        args.dataset, args.num_clients, args.train_batch_size, args.test_batch_size, args.seed
    )
    client_models, server_model = model_util.prepare_models(
        args.dataset, args.num_clients, client_devices, server_device, args.dtype
    )
    clients = setup_clients(args.num_clients, client_models, train_loaders, args.lr)
    server = FedASLServer(
        clients,
        server_device,
        lr=args.lr,
        server_model=server_model,
        server_criterion=torch.nn.CrossEntropyLoss(),
        server_accuracy_func=util.accuracy,
        num_sample_clients=args.num_sample_clients,
        local_update_steps=args.local_update,
    )

    with tqdm(total=args.iterations, desc="Training:") as t:
        for ite in range(args.iterations):
            step_loss, step_accuracy = server.train_one_step()
            t.set_postfix({"Loss": step_loss, "Accuracy": step_accuracy})
            t.update(1)

            # eval
            if args.eval_iterations != 0 and (ite + 1) % args.eval_iterations == 0:
                eval_loss, eval_accuracy = server.eval_model(test_loader, ite)
