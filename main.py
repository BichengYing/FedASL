import argparse
import numpy as np
import logging
import torch
from tqdm import tqdm

from FedASL import data_util
from FedASL import model_util
from FedASL import util
from FedASL import client as fl_client
from FedASL import server as fl_server

client_class_map = {
    "fedavg": fl_client.FedAvgClient,
    "fedasl": fl_client.FedASLClient,
    "fedgproj": fl_client.FedGaussianProjClient,
    "fedzo": fl_client.FedZOClient,
    "scaffold": fl_client.ScaffoldClient,
}
server_class_map = {
    "fedavg": fl_server.FedAvgServer,
    "fedasl": fl_server.FedASLServer,
    "fedgproj": fl_server.FedGaussianProjServer,
    "fedzo": fl_server.FedZOServer,
    "scaffold": fl_server.ScaffoldServer,
}


def setup_clients(
    num_clients: int,
    client_models: list[torch.nn.Module],
    train_loaders: list[torch.utils.data.DataLoader],
    client_devices: list[torch.device],
    method: str,
) -> list[fl_client.FedClientBase]:
    assert len(train_loaders) == num_clients
    client_class = client_class_map.get(method.lower())

    clients = []
    for i in range(args.num_clients):
        client = client_class(
            client_models[i],
            train_loaders[i],
            criterion=torch.nn.CrossEntropyLoss(),
            accuracy_func=util.accuracy,
            device=client_devices[i],
        )
        clients.append(client)

    return clients


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedASL trainning")
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--iterations", type=int, default=1e4, help="Number of iterations")
    parser.add_argument("--num-clients", type=int, default=16)
    parser.add_argument("--num_sample_clients", type=int, default=4)
    parser.add_argument("--local_update", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="mnist", help="[mnist, fashion, cifar10]")
    parser.add_argument("--seed", type=int, default=99, help="random seed")
    parser.add_argument("--dtype", type=str, default="float32", help="random seed")
    parser.add_argument("--arb_client_sampling", action="store_true", default=False)
    parser.add_argument("--eval-iterations", type=int, default=20)
    parser.add_argument(
        "--method", type=str, default="fedgproj", help="[fedasl, fedavg, fedgproj, fedzo, scaffold]"
    )

    # Per method specified args
    parser.add_argument("--num_pert", type=int, default=10)
    parser.add_argument("--same_seed", action="store_true", default=False)

    args = parser.parse_args()
    if args.method.lower() not in client_class_map.keys():
        raise ValueError(f"--method in args must be one of {list(client_class_map.keys())}")

    server_device, client_devices = data_util.auto_select_devices(args.num_clients)
    train_loaders, test_loader = data_util.prepare_dataloaders(
        args.dataset, args.num_clients, args.train_batch_size, args.test_batch_size, args.seed
    )
    client_models, server_model = model_util.prepare_models(
        args.dataset, args.num_clients, client_devices, server_device, args.dtype
    )
    clients = setup_clients(
        args.num_clients, client_models, train_loaders, client_devices, args.method
    )
    server_class = server_class_map.get(args.method.lower())
    kwargs = {}
    if args.method == "fedgproj":
        kwargs["num_pert"] = args.num_pert
    if args.method == "fedzo":
        kwargs["num_pert"] = args.num_pert
        kwargs["same_seed"] = bool(args.same_seed)
    print(kwargs)

    server = server_class(
        clients,
        server_device,
        server_model=server_model,
        server_criterion=torch.nn.CrossEntropyLoss(),
        server_accuracy_func=util.accuracy,
        num_sample_clients=args.num_sample_clients,
        local_update_steps=args.local_update,
        **kwargs,
    )
    sampling_str = "arbitrary" if args.arb_client_sampling else "uniform"
    logging.basicConfig(
        filename=f"{args.method}_{sampling_str}_{args.dataset}.log",
        level=logging.DEBUG,
        format="%(asctime)s, %(message)s",
    )
    logging.info("Iteration, Mode, loss, accuracy")

    with tqdm(total=args.iterations, desc="Training:") as t:
        np.random.seed(args.seed)
        if args.arb_client_sampling:
            sampling_prob = np.random.rand(args.num_clients) + 0.05
            sampling_prob /= sum(sampling_prob)
        else:
            sampling_prob = np.ones(args.num_clients) / args.num_clients
        print(f"{sampling_prob=}")
        for ite in range(args.iterations):
            step_loss, step_accuracy = server.train_one_step(args.lr, sampling_prob)
            t.set_postfix({"Loss": step_loss, "Accuracy": step_accuracy})
            logging.info("%d, %s, %f, %f", ite, "train", step_loss, step_accuracy)
            t.update(1)

            # eval
            if args.eval_iterations != 0 and (ite + 1) % args.eval_iterations == 0:
                eval_loss, eval_accuracy = server.eval_model(test_loader, ite)
                logging.info("%d, %s, %f, %f", ite, "eval", eval_loss, eval_accuracy)
