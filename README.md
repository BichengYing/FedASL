# Federated Optimization with exact Convergence via pUsh-pull Strategy (FOCUS)

The implementation of `FOCUS`, along with several other common FL algorithms, is available in the `FedASL` directory (`FOCUS`'s former name). The remaining files in the repository provide utility functions for data loading, model management, and other supporting tasks. The core logic is implemented in `client.py` and `server.py`, while the entry point of the repository is `main.py` located in the root directory.

An example command for running experiment are
```bash
python main.py --seed=66 --iterations=10000 --dataset=cifar10 --method=fedasl --lr=2e-3 --dirichlet-alpha=0.05 --num-clients=32 --participation=bern;
```