# Federated Optimization with exact Convergence via pUsh-pull Strategy (FOCUS) - NeurIPS'25
Arxiv Link: [https://arxiv.org/abs/2503.20117](https://arxiv.org/abs/2503.20117). 

The implementation of `FOCUS`, along with several other common FL algorithms, is available in the `FedASL` directory (`FOCUS`'s former name). The remaining files in the repository provide utility functions for data loading, model management, and other supporting tasks. The core logic is implemented in `client.py` and `server.py`, while the entry point of the repository is `main.py` located in the root directory.

An example command for running experiment are
```bash
python main.py --seed=66 --iterations=10000 --dataset=cifar10 --method=fedasl --lr=2e-3 --dirichlet-alpha=0.05 --num-clients=32 --participation=bern;
```
## Citation

```
@inproceedings{
  Ying2025exact,
  title={Exact and Linear Convergence for Federated Learning under Arbitrary Client Participation is Attainable},
  author={Bicheng Ying and Zhe Li and Haibo Yang},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=TeocEZCWnr}
}
```
