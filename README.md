# e3ferminet
E(3)-equivariant neural network ansatz for molecular VMC calculations

Final project for 6.s966 Symmetry for Machine Learning Spring 2023 by Zhening Li and Alec Zhu.

Check out our Jupyter notebook [`e3ferminet.ipynb`](e3ferminet.ipynb).

Main experiments were conducted using the training script [`train_mol.py`](train_mol.py), which uses the model implemented in [`e3ferminet_mol.py`](e3ferminet_mol.py).
Example usage:
```
python train_mol.py --config_path expt_configs/BH3_ferminet_scalar.json --save_path checkpoints/BH3_ferminet_scalar.pkl --wandb
```
Setting the flag `--wandb` allows you to track training progress using WandB.
Electron density plots are generated with [`graphs.ipynb`](graphs.ipynb).

Other architectures that we have experimented with are in [`e3ferminet.py`](e3ferminet.py) and the corresponding training script is [`train.py`](train.py).
These models only work for atoms. Example usage:
```
python train.py --config_path expt_configs/He_toy.json --save_path checkpoints/He_toy.pkl --wandb
```
Note that the toy model only works for H and He.

## Dependencies:
- JAX
- e3nn-jax
- Flax
- Optax
- Pandas
- MatPlotLib
- Plotly
- WandB
