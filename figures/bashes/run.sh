# !/usr/bin/bash

# markov
python main.py --seed=66 --iterations=10000 --dataset=cifar10 --method=fedavg --lr=2e-3 --local-update=3 --dirichlet-alpha=0.05 --arb-client-sampling --parti=markov --num-clients=32;
python main.py --seed=66 --iterations=10000 --dataset=cifar10 --method=fedasl --lr=2e-3 --local-update=3 --dirichlet-alpha=0.05 --arb-client-sampling --parti=markov --num-clients=32;
python main.py --seed=66 --iterations=10000 --dataset=cifar10 --method=scaffold --lr=2e-3 --local-update=3 --dirichlet-alpha=0.05 --arb-client-sampling --parti=markov --num-clients=32;
python main.py --seed=66 --iterations=10000 --dataset=cifar10 --method=fedau --lr=2e-3 --local-update=3 --dirichlet-alpha=0.05 --arb-client-sampling --parti=markov --num-clients=32;

# Uniform
python main.py --seed=66 --iterations=10000 --dataset=cifar10 --method=fedasl --lr=2e-3 --local-update=3 --dirichlet-alpha=0.05 --num-clients=32 --num-sample-clients=4 --parti=uniform;
python main.py --seed=66 --iterations=10000 --dataset=cifar10 --method=fedavg --lr=2e-3 --local-update=3 --dirichlet-alpha=0.05 --num-clients=32 --num-sample-clients=4 --parti=uniform;
python main.py --seed=66 --iterations=10000 --dataset=cifar10 --method=fedau --lr=2e-3 --local-update=3 --dirichlet-alpha=0.05 --num-clients=32 --num-sample-clients=4 --parti=uniform;
python main.py --seed=66 --iterations=10000 --dataset=cifar10 --method=scaffold --lr=2e-3 --local-update=3 --dirichlet-alpha=0.05 --num-clients=32 --num-sample-clients=4 --parti=uniform;
