import argparse
import json
import pickle as pkl
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, jacobian
import flax
import optax
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.express as px

import e3nn_jax as e3nn  # import e3nn-jax

from e3ferminet_mol import Ansatz

jnp.set_printoptions(precision=4, suppress=True)

print(jax.__version__)
print(flax.__version__)
print(optax.__version__)
print(e3nn.__version__)
print(jnp.ones(()).device())


class E3FerminetMol:
    def __init__(self, config):
        self.use_wandb = config.get("use_wandb", False)

        self.Zs = config["Zs"]  # length M
        self.nuclei_coords = jnp.array(config["nuclei_coords"])  # (M, 3)
        self.N_up = config.get("N_up", 1)
        self.N_down = config.get("N_down", 1)
        self.sampler = config.get("sampler")  # use M-H if None
        self.sampling_dist = config.get("sampling_dist")  # use M-H if None
        assert((self.sampler is None) == (self.sampling_dist is None))
        self.N_samples = config.get("batch_size", 20000)
        self.test_N_samples = config.get("batch_size", 50000)
        self.num_batches = config.get("num_batches", 1000)
        self.lr = config.get("lr", 0.1)
        if isinstance(self.lr, dict):
            self.lr = optax.warmup_cosine_decay_schedule(**self.lr)
        self.validate_every = config.get("validate_every", 2000)
        self.patience = config.get("patience", 200)
        self.random_key = jax.random.PRNGKey(config.get("random_seed", 0))

        self.ansatz = Ansatz(self.nuclei_coords, self.N_up, self.N_down, config["ansatz"])

        self.w = None
        self.w_list = None
        self.energies = None

        @jit
        def local_kinetic_energy(w, coords):  # coords must be unbatched
            def laplacian(coords):
                return jnp.einsum('ii->', jacobian(jacobian(self.ansatz.wavefunction, argnums=1), argnums=1)(w, coords))
            return -0.5 * laplacian(coords) / self.ansatz.wavefunction(w, coords)
        self._local_kinetic_energy = local_kinetic_energy

        @jit
        def local_potential_energy(w, coords):  # coords must be unbatched
            expanded_coords = coords.reshape((-1, 1, 3))
            rel_positions = expanded_coords - self.nuclei_coords  # (N, M, 3)
            V_e_p = -jnp.sum(jnp.array(self.Zs) / jnp.linalg.norm(rel_positions, axis=2))
            coords = coords.reshape((-1, 3))
            relative_dists = jnp.linalg.norm(jnp.expand_dims(coords, axis=0) - jnp.expand_dims(coords, axis=1), axis=2)
            V_e_e = jnp.sum(1.0 / jnp.where(relative_dists == 0.0, np.inf, relative_dists)) / 2
            return V_e_p + V_e_e
        self._local_potential_energy = local_potential_energy

        @jit
        def local_energy(w, coords):  # coords must be unbatched
            return local_kinetic_energy(w, coords) + local_potential_energy(w, coords)
        self._local_energy = local_energy

        @jit
        def energy(w, coords_batch):
            # If sampling_dist is None, assume sampling from wavefunction
            local_energies = vmap(local_energy, in_axes=(None, 0))(w, coords_batch)
            if self.sampling_dist is None:
                return jnp.mean(local_energies)
            psi = self.ansatz.wavefunction(w, coords_batch)
            scaled_probs = psi ** 2 / vmap(self.sampling_dist)(coords_batch)
            return jnp.dot(scaled_probs, local_energies) / jnp.sum(scaled_probs)
        self._energy = energy

        if self.sampler is None:
            self.MH_stdev = config["MH"].get("stdev", 0.2)
            self.MH_warmup = config["MH"].get("warmup", 500)
            self.MH_interval = config["MH"].get("interval", 10)
            self.MH_batch_size = config["MH"].get("batch_size", 64)
            self.sampled_coords = None
            def sampler(random_key, num_samples: int, N, Zs: list[int], nuclei_coords):
                # returns jnp array of shape (num_samples, 3*Z) sampled from the wavefunction
                if self.sampled_coords is None:
                    warmup = self.MH_warmup
                    random_key, subkey = jax.random.split(random_key)
                    # assign electrons to atoms
                    num_electrons = [0 for _ in range(len(Zs))]
                    electrons_left = N
                    while electrons_left > 0:
                        new_electrons_left = electrons_left
                        for i in range(len(Zs)):
                            if num_electrons[i] < Zs[i]:
                                num_electrons[i] += 1
                                new_electrons_left -= 1
                                if new_electrons_left == 0:
                                    break
                        if electrons_left == new_electrons_left:
                            break
                        electrons_left = new_electrons_left
                    if electrons_left > 0:
                        while electrons_left > 0:
                            for i in range(len(Zs)):
                                num_electrons[i] += 1
                                electrons_left -= 1
                                if electrons_left == 0:
                                    break
                    coords_centers = jnp.concatenate([nuclei_coords[i] for i in range(len(Zs)) for _ in range(num_electrons[i])])
                    self.sampled_coords = self.MH_stdev * jax.random.normal(subkey, (self.MH_batch_size, 3*N)) + coords_centers
                else:
                    warmup = self.MH_interval
                coords = []
                num_iters = warmup + (num_samples - 1) // self.MH_batch_size + 1
                num_coords_remaining = num_samples
                for i in range(num_iters):
                    random_key, subkey = jax.random.split(random_key)
                    proposal_coords = self.sampled_coords + self.MH_stdev * jax.random.normal(subkey, (self.MH_batch_size, 3*N))
                    acceptance_ratios = (self.ansatz.wavefunction(self.w, proposal_coords) / self.ansatz.wavefunction(self.w, self.sampled_coords)) ** 2
                    random_key, subkey = jax.random.split(random_key)
                    self.sampled_coords = jnp.where(np.expand_dims(jax.random.uniform(subkey, (self.MH_batch_size,)) < acceptance_ratios, axis=1),
                                                    proposal_coords,
                                                    self.sampled_coords)
                    if i >= warmup:
                        if self.MH_batch_size <= num_coords_remaining:
                            coords_to_add = self.sampled_coords
                            num_coords_remaining -= self.MH_batch_size
                        else:
                            coords_to_add = self.sampled_coords[:num_coords_remaining]
                            num_coords_remaining = 0
                        coords.append(coords_to_add)
                return jnp.concatenate(coords)
            self.sampler = sampler
        else:
            assert "MH" not in config
    
    def init_weights(self):
        self.random_key, subkey = jax.random.split(self.random_key)
        self.w = self.ansatz.init_weights(subkey)
        # print("WEIGHTS:", self.w)
        # print("ENERGY:", self._energy(self.w, coords_batch))
        # if self.regularize:
        #     print("REGULARIZED ENERGY:", self._regularized_energy(self.w, coords_batch))

    def load_weights(self, ckpt_path):
        self.w = self.ansatz.load_weights(ckpt_path)

    def save_weights(self, save_path):
        with open(save_path, 'wb') as f:
            pkl.dump(self.w, f)
    
    def train_loop(self):
        # Training loop

        self.init_weights()

        grad_energy = jit(grad(self._energy))

        optimizer = optax.adamw(learning_rate=self.lr)
        opt_state = optimizer.init(self.w)

        weights = [self.w]
        self.random_key, subkey = jax.random.split(self.random_key)
        coords_batch = self.sampler(subkey, self.N_samples, self.N_up + self.N_down, self.Zs, self.nuclei_coords)
        loss = self._energy(self.w, coords_batch)
        losses = [loss]
        energy = self.test()
        self.energies = [energy]
        for step in tqdm(range(self.num_batches)):
            self.random_key, subkey = jax.random.split(self.random_key)
            coords_batch = self.sampler(subkey, self.N_samples, self.N_up + self.N_down, self.Zs, self.nuclei_coords)
            grads = grad_energy(self.w, coords_batch)
            updates, opt_state = optimizer.update(grads, opt_state, self.w)
            self.w = optax.apply_updates(self.w, updates)
            loss = self._energy(self.w, coords_batch)
            losses.append(loss)
            if step % self.validate_every == 0:
                energy = self.test()
                self.energies.append(energy)
                print("ENERGY:", energy)
                if self.use_wandb:
                    wandb.log({"loss": loss, "energy": energy})
                weights.append(self.w)
            else:
                if self.use_wandb:
                    wandb.log({"loss": loss})
            if self.patience is not None and step - np.argmin(self.energies) * self.validate_every >= self.patience:
                break
        self.w_list = weights
    
    def choose_weights(self, idx):
        if idx == "best":
            idx = np.argmin(self.energies)
            print(f"BEST INDEX: {idx}")
        elif idx == "last":
            idx = -1
        self.w = self.w_list[idx]

    def test(self, test_N_samples=None):
        if test_N_samples is None:
            test_N_samples = self.test_N_samples
        self.random_key, subkey = jax.random.split(self.random_key)
        coords_batch = self.sampler(subkey, test_N_samples, self.N_up + self.N_down, self.Zs, self.nuclei_coords)
        test_energy = self._energy(self.w, coords_batch)
        # print("GROUND STATE ENERGY: {:.4f}".format(self._energy(self.w, coords_batch)))
        return test_energy
    
    def plot_one_electron_radial(self, max_r, plot_samples=5000):
        radii = jnp.linspace(0, max_r, plot_samples+1)
        coords_batch = np.hstack((np.expand_dims(radii, axis=1), np.zeros((plot_samples+1, 3*self.Z - 1))))
        psi = self.ansatz.wavefunction(self.w, coords_batch)
        x_label = "$r$"
        y_label = "$\\psi(r\\hat e_z, 0, \\ldots, 0)$"
        df = pd.DataFrame({x_label: radii, y_label: psi})
        fig = px.line(df, x=x_label, y=y_label)
        fig.show()

    def plot_density_3D(self, plot_samples=5000):
        self.random_key, subkey = jax.random.split(self.random_key)
        coords_batch = self.sampler(subkey, self.Z, plot_samples)
        densities = vmap(self.ansatz.wavefunction)(self.w, coords_batch) ** 2
        max_density = jnp.max(densities)
        self.random_key, subkey = jax.random.split(self.random_key)
        coords_batch = coords_batch[max_density * jax.random.uniform(subkey, shape=(plot_samples,)) < densities]
        df = pd.DataFrame(coords_batch.reshape((-1, 3)), columns=['x', 'y', 'z'])
        print(df.head())
        fig = px.scatter_3d(df, x='x', y='y', z='z')
        fig.show()
    
    def plot_density_2D(self, pixel_size=0.01, step_size=0.1):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--load_path")
    parser.add_argument("--save_path")
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config["use_wandb"] = args.wandb

    if args.wandb:
        wandb.init(project="e3ferminet", config=config)

    atom_model = E3FerminetMol(config)
    if args.load_path is not None:
        atom_model.load_weights(args.load_path)
    else:
        atom_model.train_loop()
        atom_model.choose_weights("best")
    atom_model.sampled_coords = None
    test_energy = atom_model.test()
    print("ENERGY: {:.4f}".format(test_energy))
    atom_model.plot_one_electron_radial(4)
    if args.save_path is not None:
        atom_model.save_weights(args.save_path)
