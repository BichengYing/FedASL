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
    "fedau": fl_client.FedAUClient,
}
server_class_map = {
    "fedavg": fl_server.FedAvgServer,
    "fedasl": fl_server.FedASLServer,
    "fedgproj": fl_server.FedGaussianProjServer,
    "fedzo": fl_server.FedZOServer,
    "scaffold": fl_server.ScaffoldServer,
    "fedau": fl_server.FedAUServer,
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
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--iterations", type=int, default=1e4, help="Number of iterations")
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--num-sample-clients", type=int, default=10)
    parser.add_argument("--local-update", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="mnist", help="[mnist, fashion, cifar10]")
    parser.add_argument("--seed", type=int, default=66, help="random seed")
    parser.add_argument("--dtype", type=str, default="float32", help="random seed")
    parser.add_argument("--eval-iterations", type=int, default=25)
    parser.add_argument(
        "--method",
        type=str,
        default="fedavg",
        help="[fedasl, fedavg, fedgproj, fedzo, scaffold, fedau]",
    )
    parser.add_argument("--iid", action="store_true", default=False)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.1)
    parser.add_argument(
        "--participation", type=str, default="bern", help=["bern", "markov", "uniform"]
    )

    # Per method specified args
    parser.add_argument("--num-pert", type=int, default=10)
    parser.add_argument("--same-seed", action="store_true", default=False)

    args = parser.parse_args()
    if args.method.lower() not in client_class_map.keys():
        raise ValueError(f"--method in args must be one of {list(client_class_map.keys())}")

    server_device, client_devices = data_util.auto_select_devices(args.num_clients)
    if args.iid:  # IID
        train_loaders, test_loader = data_util.prepare_dataloaders(
            args.dataset, args.num_clients, args.train_batch_size, args.test_batch_size, args.seed
        )
    else:  # Non-IID
        train_loaders, test_loader = data_util.prepare_dataloaders(
            args.dataset,
            args.num_clients,
            args.train_batch_size,
            args.test_batch_size,
            args.seed,
            args.dirichlet_alpha,
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
    logging.basicConfig(
        filename=f"{args.method}_{args.dataset}_{args.lr}_{args.participation}_{args.local_update}.log",
        level=logging.DEBUG,
        format="%(asctime)s, %(message)s",
    )
    logging.info("Iteration, Mode, loss, accuracy")

    with tqdm(total=args.iterations, desc="Training:") as t:
        np.random.seed(args.seed)
        if args.participation == "bern":
            client_probabilities_low = np.random.uniform(0.1, 0.3, size=int(args.num_clients * 0.9))
            client_probabilities_high = np.random.uniform(
                0.1, 0.9, size=int(args.num_clients * 0.1)
            )
            client_probabilities = np.concatenate(
                [client_probabilities_low, client_probabilities_high]
            )
        elif args.participation == "markov":
            client_probabilities = np.random.uniform(0.1, 0.9, size=int(args.num_clients))
            transition_matrix = np.array([[0.8, 0.2], [0.8, 0.2]])
            current_states = np.random.binomial(1, client_probabilities)
        for ite in range(args.iterations):
            # Arbitrary sampling
            if args.participation == "bern":
                sampling_prob = np.random.binomial(1, client_probabilities)
            elif args.participation == "markov":
                sampling_prob = []
                for i in range(args.num_clients):
                    current_state = current_states[i]
                    next_state = np.random.choice([0, 1], p=transition_matrix[current_state])
                    current_states[i] = next_state
                    sampling_prob.append(next_state)
            elif args.participation == "uniform":
                sampling_prob = np.ones(args.num_clients) / args.num_clients
            else:
                raise ValueError(f"Unknown {args.participation=}")
            step_loss, step_accuracy = server.train_one_step(
                args.lr, sampling_prob, args.participation
            )
            t.set_postfix({"Loss": step_loss, "Accuracy": step_accuracy})
            logging.info("%d, %s, %f, %f", ite, "train", step_loss, step_accuracy)
            t.update(1)

            # Evaluation
            if args.eval_iterations != 0 and (ite + 1) % args.eval_iterations == 0:
                eval_loss, eval_accuracy = server.eval_model(test_loader, ite)
                logging.info("%d, %s, %f, %f", ite, "eval", eval_loss, eval_accuracy)
