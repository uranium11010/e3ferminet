import pickle as pkl
import jax
from jax import jit
import jax.numpy as jnp
import e3nn_jax as e3nn  # import e3nn-jax


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


def max_l_from_Z(Z):
    l = 0
    maxZ = 4
    while maxZ < Z:
        l += 1
        maxZ += 4 * (l + 1)**2
    return l


class FerminetAnsatz:
    def __init__(self, Z, N_up, N_down, config):
        self.Z = Z
        self.N_up = N_up
        self.N_down = N_down
        self.N = N_up + N_down

        self.channels_per_irrep = config.get("channels_per_irrep", 16)
        self.lmax = config.get("lmax", max_l_from_Z(Z))
        self.hidden_irreps = [f"{self.channels_per_irrep}x{l}{'o' if l % 2 else 'e'}" for l in range(self.lmax + 1)]
        self.hidden_irreps_before_gate = self.hidden_irreps.copy()
        self.hidden_irreps_before_gate[0] = str((self.lmax + 1) * self.channels_per_irrep) + "x0e"

        def concat_distances(x):
            abs_dists = jnp.expand_dims(jnp.linalg.norm(x, axis=-1), axis=-1)
            abs_dists = e3nn.IrrepsArray("0e", abs_dists)
            rel_dists = jnp.linalg.norm(jnp.expand_dims(x, axis=-2) - jnp.expand_dims(x, axis=-3), axis=-1)
            rel_dists = e3nn.IrrepsArray(f"{self.N}x0e", rel_dists)
            x_irreps = e3nn.IrrepsArray(f"1o", x)
            return e3nn.concatenate([x_irreps, abs_dists, rel_dists])
        self.concat_distances = concat_distances

        self.num_layers = config.get("num_layers", 4)
        self.layers = []
        for i in range(self.num_layers):
            layer = {
                    "averaging": lambda h: e3nn.concatenate([e3nn.mean(h[...,self.N_up:,:], axis=-2, keepdims=True),
                        e3nn.mean(h[...,:self.N_up,:], axis=-2, keepdims=True)]),
                    "concat": lambda h, avg: e3nn.concatenate([h, e3nn.concatenate([avg] * self.N, axis=-2)]),
                    # "append_one": lambda avg: e3nn.concatenate([avg, e3nn.IrrepsArray("0e", jnp.ones(avg.shape[:-1] + (1,)))]),
                    # "tensor": lambda h_avg, avg_1: e3nn.tensor_product(h_avg, avg_1, filter_ir_out=e3nn.Irrep.iterator(lmax=self.lmax)),
                    "tensor": lambda h_avg: e3nn.tensor_square(h_avg).filter(keep=e3nn.Irrep.iterator(lmax=self.lmax)),
                    "tensor_residual": lambda t, h: e3nn.concatenate([t, h]),
                    "linear": e3nn.flax.Linear('+'.join(self.hidden_irreps_before_gate), biases=True),
                    # "gate": lambda f: e3nn.gate(f, even_act=jax.nn.tanh, even_gate_act=jax.nn.tanh),
                    "gate": e3nn.gate,
            }
            if i > 0:
                layer["residual"] = lambda g, h: g + h
            self.layers.append(layer)

        self.linear_head = e3nn.flax.Linear(irreps_out=f"{self.N}x0e", biases=True)

        self.envelope = lambda zeta, x: jnp.exp(-zeta * jnp.linalg.norm(x, axis=-1))

        @jit
        def wavefunction(w, coords):  # coords can be unbatched or batched
            x = coords.reshape(coords.shape[:-1] + (self.N, 3))
            h = self.concat_distances(x)
            for i, layer in enumerate(self.layers):
                avg = layer["averaging"](h)
                h_avg = layer["concat"](h, avg)
                # avg_1 = layer["append_one"](avg)
                # t = layer["tensor"](h_avg, avg_1)
                # z = layer["linear"].apply(w["linear_hidden"][i], t)
                t = layer["tensor"](h_avg)
                r = layer["tensor_residual"](t, h)
                z = layer["linear"].apply(w["linear_hidden"][i], r)
                g = layer["gate"](z)
                if i == 0:
                    h = g
                else:
                    h = layer["residual"](g, h)
            z = self.linear_head.apply(w["linear_head"], x).array
            e = self.envelope(jnp.abs(w["envelope"]), x)
            phi = z * jnp.expand_dims(e, axis=-1)
            D_up = jnp.linalg.det(phi[...,:self.N_up,:self.N_up])
            D_down = jnp.linalg.det(phi[...,self.N_up:,self.N_up:])
            return D_up * D_down
        self.wavefunction = wavefunction

    def init_weights(self, random_key):
        subkeys = jax.random.split(random_key, num=self.num_layers+2)
        w = {"linear_hidden": []}
        coords = jnp.empty(3 * self.N)
        x = coords.reshape(coords.shape[:-1] + (self.N, 3))
        h = self.concat_distances(x)
        print("h:", h.shape, h.irreps)
        for i, layer in enumerate(self.layers):
            avg = layer["averaging"](h)
            print("avg:", avg.shape, avg.irreps)
            h_avg = layer["concat"](h, avg)
            print("h_avg:", h_avg.shape, h_avg.irreps)
            # avg_1 = layer["append_one"](avg)
            # print("avg_1:", avg_1.shape, avg_1.irreps)
            # t = layer["tensor"](h_avg, avg_1)
            # print("t:", t.shape, t.irreps)
            # w["linear_hidden"].append(layer["linear"].init(subkeys[i], t))
            # z = layer["linear"].apply(w["linear_hidden"][i], t)
            # print("z:", z.shape, z.irreps)
            t = layer["tensor"](h_avg)
            print("t:", t.shape, t.irreps)
            r = layer["tensor_residual"](t, h)
            print("r:", r.shape, r.irreps)
            w["linear_hidden"].append(layer["linear"].init(subkeys[i], r))
            z = layer["linear"].apply(w["linear_hidden"][i], r)
            print("z:", z.shape, z.irreps)
            g = layer["gate"](z)
            print("g:", g.shape, g.irreps)
            if i == 0:
                h = g
            else:
                h = layer["residual"](g, h)
            print("h:", h.shape, h.irreps)
        w["linear_head"] = self.linear_head.init(subkeys[self.num_layers], x)
        z = self.linear_head.apply(w["linear_head"], x).array
        print("z:", z.shape)
        w["envelope"] = jnp.sqrt(jax.random.chisquare(subkeys[self.num_layers+1], shape=(self.N,), df=2) * self.Z)
        e = self.envelope(jnp.abs(w["envelope"]), x)
        print("e:", e.shape)
        phi = z * jnp.expand_dims(e, axis=-1)
        print("phi:", phi.shape)
        D_up = jnp.linalg.det(phi[...,:self.N_up,:self.N_up])
        print("D_up:", D_up.shape)
        D_down = jnp.linalg.det(phi[...,self.N_up:,self.N_up:])
        print("D_down:", D_down.shape)
        return w

    def load_weights(self, ckpt_path):
        with open(ckpt_path, 'rb') as f:
            return pkl.load(f)



ANSATZE = {
        "toy": ToyAnsatz,
        "manual": ManualAnsatz,
        "ferminet": FerminetAnsatz
}

def Ansatz(Z, N_up, N_down, config):
    return ANSATZE[config.get("type", "ferminet")](Z, N_up, N_down, config)


if __name__ == "__main__":
    for Z in range(1, 120):
        print(Z, max_l_from_Z(Z))
