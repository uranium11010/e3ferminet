import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, jacobian
import flax
import optax
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.express as px

import e3nn_jax as e3nn  # import e3nn-jax

jnp.set_printoptions(precision=4, suppress=True)

print(jax.__version__)
print(flax.__version__)
print(optax.__version__)
print(e3nn.__version__)
print(jnp.ones(()).device())

class ToyAnsatz:
    def __init__(self, Z, N_up, N_down, config):
        self.Z = Z
        self.N_up = N_up
        self.N_down = N_down

        self.mlp = e3nn.flax.MultiLayerPerceptron([5, 5, 5, 1], act=jax.nn.gelu, output_activation=jax.nn.sigmoid)
        self.envelope = lambda zeta, coords: jnp.exp(-zeta * jnp.linalg.norm(coords, axis=-1))

        @jit
        def wavefunction(w, coords):  # coords can be unbatched or batched
            # TO-DO antisymmetrize the wavefunction at the end (spin-up and spin-down separately)
            x = e3nn.tensor_square(e3nn.IrrepsArray(f"{self.Z}x1o", coords)).filter(keep="0e")
            return self.mlp.apply(w["mlp"], x.array).squeeze(-1) * self.envelope(jnp.abs(w["envelope"]), coords)
        self.wavefunction = wavefunction

    def init_weights(self, random_key):  # coords can be batched or unbatched
        subkey1, subkey2 = jax.random.split(random_key)
        coords = jnp.empty((3 * self.Z,))
        x = e3nn.tensor_square(e3nn.IrrepsArray(f"{self.Z}x1o", coords)).filter(keep="0e")
        return {
            "mlp": self.mlp.init(subkey1, x),
            "envelope": jnp.sqrt(jax.random.chisquare(subkey2, df=2)) * self.Z
        }


class ManualAnsatz:
    def __init__(self, Z, N_up, N_down, config):
        self.Z = Z
        self.N_up = N_up
        self.N_down = N_down
        self.N = N_up + N_down

        self.hidden_irreps = "5x0e+5x1e+5x1o+5x2e+5x2o"
        self.hidden_irreps_before_gate = "25x0e+5x1e+5x1o+5x2e+5x2o"
        self.lmax = 2
        # assert self.lmax + 1 == len(self.hidden_channels)

        self.tensor = lambda input: e3nn.tensor_square(input).filter(keep=e3nn.Irrep.iterator(lmax=self.lmax))
        self.linear1 = e3nn.flax.Linear(irreps_out=self.hidden_irreps_before_gate, biases=True)
        self.linear2 = e3nn.flax.Linear(irreps_out=self.hidden_irreps_before_gate, biases=True)
        self.linear_head = e3nn.flax.Linear(irreps_out="0e", biases=True)
        self.envelope = lambda zeta, coords: jnp.exp(-zeta * jnp.linalg.norm(coords, axis=-1))

        @jit
        def wavefunction(w, coords):  # coords can be unbatched or batched
            coords_irreps = e3nn.IrrepsArray(f"{self.N}x1o", coords)
            x = self.tensor(coords_irreps)
            x = self.linear1.apply(w["linear1"], x)
            x = e3nn.gate(x)
            x = self.linear2.apply(w["linear2"], x)
            x = e3nn.gate(x)
            x = x.filter(keep="0e")
            x = self.linear_head.apply(w["linear_head"], x).array.squeeze(-1)
            return x * self.envelope(jnp.abs(w["envelope"]), coords)
        self.wavefunction = wavefunction

    def init_weights(self, random_key):  # coords can be batched or unbatched
        w = {}
        subkey1, subkey2, subkey3, subkey4 = jax.random.split(random_key, num=4)
        coords = jnp.empty((3 * self.Z,))
        coords_irreps = e3nn.IrrepsArray(f"{self.N}x1o", coords)
        x = self.tensor(coords_irreps)
        w["linear1"] = self.linear1.init(subkey1, x)
        x = self.linear1.apply(w["linear1"], x)
        print(x.irreps)
        x = e3nn.gate(x)
        w["linear2"] = self.linear2.init(subkey2, x)
        x = self.linear2.apply(w["linear2"], x)
        x = e3nn.gate(x)
        x = x.filter(keep="0e")
        w["linear_head"] = self.linear_head.init(subkey3, x)
        x = self.linear_head.apply(w["linear_head"], x).array.squeeze(-1)
        w["envelope"] = jnp.sqrt(jax.random.chisquare(subkey4, df=2)) * self.Z
        return w


class FerminetAnsatz:
    def __init__(self, Z, N_up, N_down, config):
        self.Z = Z
        self.N_up = N_up
        self.N_down = N_down
        self.N = N_up + N_down

        # self.hidden_irreps = "5x0e+5x1e+5x1o+5x2e+5x2o"
        # self.hidden_irreps_before_gate = "25x0e+5x1e+5x1o+5x2e+5x2o"
        self.lmax = config.get("lmax", )

        self.tensor = lambda input: e3nn.tensor_square(input).filter(keep=e3nn.Irrep.iterator(lmax=self.lmax))
        self.linear1 = e3nn.flax.Linear(irreps_out=self.hidden_irreps_before_gate, biases=True)
        self.linear2 = e3nn.flax.Linear(irreps_out=self.hidden_irreps_before_gate, biases=True)
        self.linear_head = e3nn.flax.Linear(irreps_out="0e", biases=True)
        self.envelope = lambda zeta, coords: jnp.exp(-zeta * jnp.linalg.norm(coords, axis=-1))

        @jit
        def wavefunction(w, coords):  # coords can be unbatched or batched
            coords_irreps = e3nn.IrrepsArray(f"{self.N}x1o", coords)
            x = self.tensor(coords_irreps)
            x = self.linear1.apply(w["linear1"], x)
            x = e3nn.gate(x)
            x = self.linear2.apply(w["linear2"], x)
            x = e3nn.gate(x)
            x = x.filter(keep="0e")
            x = self.linear_head.apply(w["linear_head"], x).array.squeeze(-1)
            return x * self.envelope(jnp.abs(w["envelope"]), coords)
        self.wavefunction = wavefunction

    def init_weights(self, random_key):  # coords can be batched or unbatched
        w = {}
        subkey1, subkey2, subkey3, subkey4 = jax.random.split(random_key, num=4)
        coords = jnp.empty((3 * self.Z,))
        coords_irreps = e3nn.IrrepsArray(f"{self.N}x1o", coords)
        x = self.tensor(coords_irreps)
        w["linear1"] = self.linear1.init(subkey1, x)
        x = self.linear1.apply(w["linear1"], x)
        print(x.irreps)
        x = e3nn.gate(x)
        w["linear2"] = self.linear2.init(subkey2, x)
        x = self.linear2.apply(w["linear2"], x)
        x = e3nn.gate(x)
        x = x.filter(keep="0e")
        w["linear_head"] = self.linear_head.init(subkey3, x)
        x = self.linear_head.apply(w["linear_head"], x).array.squeeze(-1)
        w["envelope"] = jnp.sqrt(jax.random.chisquare(subkey4, df=2)) * self.Z
        return w


class E3FerminetAtom:
    def __init__(self, config):
        self.Z = config.get("Z", 1)
        self.N_up = config.get("N_up", 1)
        self.N_down = config.get("N_down", 1)
        self.sampler = config.get("sampler")  # use M-H if None
        self.sampling_dist = config.get("sampling_dist")  # use M-H if None
        assert((self.sampler is None) == (self.sampling_dist is None))
        self.N_samples = config.get("batch_size", 20000)
        self.num_batches = config.get("num_batches", 1000)
        self.lr = config.get("lr", 0.1)
        self.validate_every = config.get("validate_every", 2000)
        self.moving_avg_coeff = config.get("moving_avg_coeff", 0.1)
        self.regularize = "regularize" in config
        self.regularize_pow = config["regularize"].get("pow", 8) if self.regularize else None
        self.regularize_coeff = config["regularize"].get("coeff", 100) if self.regularize else None
        self.regularize_max_r = config["regularize"].get("max_r", 2) if self.regularize else None
        self.patience = config.get("patience", 200)
        self.random_key = jax.random.PRNGKey(config.get("random_seed", 0))

        self.ansatz = ManualAnsatz(self.Z, self.N_up, self.N_down, config["ansatz"])

        self.w = None
        self.w_list = None
        self.energy_moving_avgs = None

        @jit
        def local_kinetic_energy(w, coords):  # coords must be unbatched
            def laplacian(coords):
                return jnp.einsum('ii->', jacobian(jacobian(self.ansatz.wavefunction, argnums=1), argnums=1)(w, coords))
            return -0.5 * laplacian(coords) / self.ansatz.wavefunction(w, coords)
        self._local_kinetic_energy = local_kinetic_energy

        @jit
        def local_potential_energy(w, coords):  # coords must be unbatched
            coords = coords.reshape((-1, 3))
            V_e_p = -self.Z * jnp.sum(1 / jnp.linalg.norm(coords, axis=1), axis=0)
            relative_dists = jnp.linalg.norm(jnp.expand_dims(coords, axis=0) - jnp.expand_dims(coords, axis=1), axis=2)
            V_e_e = jnp.sum(1 / jnp.where(relative_dists == 0, np.inf, relative_dists)) / 2
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

        if self.regularize:
            @jit
            def regularized_energy(w, coords_batch):
                reshaped_coords_batch = coords_batch.reshape((coords_batch.shape[0], -1, 3))
                penalty = jnp.sum((jnp.linalg.norm(reshaped_coords_batch, axis=2) / self.regularize_max_r) ** self.regularize_pow, axis=1)
                if self.sampling_dist is None:
                    cum_penalty = jnp.mean(penalty)
                else:
                    psi = self.ansatz.wavefunction(w, coords_batch)
                    scaled_probs = psi ** 2 / vmap(self.sampling_dist)(coords_batch)
                    cum_penalty = jnp.dot(scaled_probs, penalty) / jnp.sum(scaled_probs)
                return energy(w, coords_batch) + self.regularize_coeff * cum_penalty
            self._regularized_energy = regularized_energy
        
        if self.sampler is None:
            self.MH_stdev = config["MH"].get("stdev", 0.2)
            self.MH_warmup = config["MH"].get("warmup", 500)
            self.MH_interval = config["MH"].get("interval", 10)
            self.MH_batch_size = config["MH"].get("batch_size", 64)
            self.sampled_coords = None
            def sampler(random_key, Z, num_samples):
                # returns jnp array of shape (num_samples, 3*Z) sampled from the wavefunction
                if self.sampled_coords is None:
                    warmup = self.MH_warmup
                    random_key, subkey = jax.random.split(random_key)
                    self.sampled_coords = self.MH_stdev * jax.random.normal(subkey, (self.MH_batch_size, 3*Z))
                else:
                    warmup = self.MH_interval
                coords = []
                num_iters = warmup + (num_samples - 1) // self.MH_batch_size + 1
                num_coords_remaining = num_samples
                for i in range(num_iters):
                    random_key, subkey = jax.random.split(random_key)
                    proposal_coords = self.sampled_coords + self.MH_stdev * jax.random.normal(subkey, (self.MH_batch_size, 3*Z))
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
                # self.sampled_coords = jax.random.normal(random_key, (3*Z,))
                # for i in range(500): #need time to stabilize chain, tune this number
                #     random_key, subkey = jax.random.split(random_key)
                #     proposal_coords = self.sampled_coords + 0.1*jax.random.normal(subkey, (3*Z,)) #need to tune stdev
                #     a = self.ansatz.wavefunction(self.w, proposal_coords)**2 / self.ansatz.wavefunction(self.w, x_init)**2
                #     if jax.random.uniform(subkey) < a:
                #         self.sample_coords = proposal_coords
                # coords = []
                # for i in range(num_samples):
                #     random_key, subkey = jax.random.split(random_key)
                #     proposal_coords = x_init + 0.3*jax.random.normal(subkey, (3*Z,))
                #     a = self.ansatz.wavefunction(self.w, proposal_coords)**2 / self.ansatz.wavefunction(self.w, x_init)**2
                #     if jax.random.uniform(subkey) < a:
                #         x_init = proposal_coords
                #     coords.append(x_init)
                # return jax.numpy.array(coords)
            self.sampler = sampler
        else:
            assert "MH" not in config
    
    def init_weights(self):
        self.random_key, subkey = jax.random.split(self.random_key)
        self.w = self.ansatz.init_weights(subkey)
        print("WEIGHTS:", self.w)
        # print("ENERGY:", self._energy(self.w, coords_batch))
        # if self.regularize:
        #     print("REGULARIZED ENERGY:", self._regularized_energy(self.w, coords_batch))
    
    def train_loop(self):
        # Training loop

        self.init_weights()

        grad_energy = jit(grad(self._regularized_energy)) if self.regularize else jit(grad(self._energy))

        optimizer = optax.adamw(learning_rate=self.lr)
        opt_state = optimizer.init(self.w)

        weights = [self.w]
        self.random_key, subkey = jax.random.split(self.random_key)
        coords_batch = self.sampler(subkey, self.Z, self.N_samples)
        energy = self._energy(self.w, coords_batch)
        energies = [energy]
        energy_moving_avgs = [energy]
        for step in tqdm(range(self.num_batches)):
            self.random_key, subkey = jax.random.split(self.random_key)
            coords_batch = self.sampler(subkey, self.Z, self.N_samples)
            grads = grad_energy(self.w, coords_batch)
            updates, opt_state = optimizer.update(grads, opt_state, self.w)
            self.w = optax.apply_updates(self.w, updates)
            weights.append(self.w)
            energy = self._energy(self.w, coords_batch)
            energies.append(energy)
            energy_moving_avgs.append(energy_moving_avgs[-1] * (1 - self.moving_avg_coeff) + energy * self.moving_avg_coeff)
            if step % self.validate_every == 0:
                self.test()
            if self.patience is not None and step - np.argmin(energy_moving_avgs) >= self.patience:
                break
        self.w_list = weights
        self.energy_moving_avgs = energy_moving_avgs
        learning_curve_df = pd.DataFrame({"Batch index": np.arange(len(energy_moving_avgs)), "Energy": energies})
        fig = px.line(learning_curve_df, x="Batch index", y="Energy")
        fig.show()
    
    def choose_weights(self, idx):
        if idx == "best":
            idx = jnp.argmin(self.energy_moving_avgs)
            print(f"BEST INDEX: {idx}")
        elif idx == "last":
            idx = -1
        self.w = self.w_list[idx]

    def test(self, test_N_samples=50000):
        self.random_key, subkey = jax.random.split(self.random_key)
        coords_batch = self.sampler(subkey, self.Z, test_N_samples)
        print("GROUND STATE ENERGY: {:.4f}".format(self._energy(self.w, coords_batch)))
    
    def plot_one_electron_radial(self, max_r, plot_samples=5000):
        radii = jnp.linspace(0, max_r, plot_samples+1)
        coords_batch = np.hstack((np.expand_dims(radii, axis=1), np.zeros((plot_samples+1, 3*self.Z - 1))))
        psi = self.ansatz.wavefunction(self.w, coords_batch)
        x_label = "$r$"
        y_label = "$\\\\psi(r\\\\hat e_z, 0, \\\\ldots, 0)$"
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
    max_r_hydrogen = 4
    hydrogen_config = {
        "random_seed": 1,
        "Z": 1,
        "N_up": 1,
        "N_down": 0,
        # "batch_size": 2000,
        # "num_batches": 50,
        "batch_size": 64,
        "num_batches": 1000,
        "patience": None,
        # "lr": 0.001,
        "lr": 1e-5,
        #"sampling_dist": lambda coords: 1,
        #"sampler": lambda random_key, Z, num_samples: jax.random.ball(random_key, 3, shape=(num_samples, Z)).reshape((num_samples, -1)) * max_r_hydrogen,
        "sampling_dist" : None,
        "sampler" : None,
        "moving_avg_coeff": 0.1,
        "ansatz": {},
        "MH": {
            "stdev": 0.2,
            "warmup": 500,
            "batch_size": 64
        }
        # "regularize": {
        #     "max_r": max_r_hydrogen,
        #     "pow": 8,
        #     "coeff": 1
        # }
    }

    max_r_helium = 2
    helium_config = {
        "random_seed": 0,
        "Z": 2,
        "N_up": 1,
        "N_down": 1,
        # "batch_size": 2000,
        # "num_batches": 25,
        "batch_size": 128,
        "num_batches": 100000,
        "validate_every": 2000,
        "patience": None,
        "lr": optax.warmup_cosine_decay_schedule(5e-5, 5e-4, 100, 100000, end_value=5e-6, exponent=1.0),
        # "sampling_dist": lambda coords: 1,
        # "sampler": lambda random_key, Z, num_samples: jax.random.ball(random_key, 3, shape=(num_samples, Z)).reshape((num_samples, -1)) * max_r_helium,
        "sampling_dist" : None,
        "sampler" : None,
        "ansatz": {},
        "MH": {
            "stdev": 0.2,
            "warmup": 500,
            "interval": 10,
            "batch_size": 64
        }
        # "regularize": {
        #     "max_r": max_r_helium,
        #     "regularize_pow": 8,
        #     "regularize_coeff": 0,
        # }
    }

    atom_model = E3FerminetAtom(helium_config)
    atom_model.train_loop()
    atom_model.choose_weights("last")
    atom_model.test()
    atom_model.plot_one_electron_radial(4)
