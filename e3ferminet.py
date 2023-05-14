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
            return self.mlp.apply(w["mlp"], x.array).squeeze(-1) * self.envelope(jnp.abs(w["envelope"]) + self.Z * 0.5, coords)
        self.wavefunction = wavefunction

    def init_weights(self, random_key):  # coords can be batched or unbatched
        subkey1, subkey2 = jax.random.split(random_key)
        coords = jax.random.normal(random_key, (3 * self.Z,))
        x = e3nn.tensor_square(e3nn.IrrepsArray(f"{self.Z}x1o", coords)).filter(keep="0e")
        return {
            "mlp": self.mlp.init(subkey1, x),
            "envelope": jnp.sqrt(jax.random.chisquare(subkey2, df=2)) * self.Z * 0.5
        }

    def load_weights(self, ckpt_path):
        with open(ckpt_path, 'rb') as f:
            return pkl.load(f)


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
        coords = jax.random.normal(random_key, (3 * self.Z,))
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

    def load_weights(self, ckpt_path):
        with open(ckpt_path, 'rb') as f:
            return pkl.load(f)


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
                "averaging": (lambda h: e3nn.mean(h, axis=-2, keepdims=True) if self.N_up == 0 or self.N_down == 0
                    else e3nn.concatenate([e3nn.mean(h[...,:self.N_up,:], axis=-2, keepdims=True),
                    e3nn.mean(h[...,self.N_up:,:], axis=-2, keepdims=True)])),
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
            z = self.linear_head.apply(w["linear_head"], h).array
            e = self.envelope(jnp.abs(w["envelope"]), x)
            phi = z * jnp.expand_dims(e, axis=-1)
            D_up = jnp.linalg.det(phi[...,:self.N_up,:self.N_up])
            D_down = jnp.linalg.det(phi[...,self.N_up:,self.N_up:])
            return D_up * D_down
        self.wavefunction = wavefunction

    def init_weights(self, random_key):
        subkeys = jax.random.split(random_key, num=self.num_layers+2)
        w = {"linear_hidden": []}
        coords = jax.random.normal(random_key, (3 * self.N,))
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
        w["linear_head"] = self.linear_head.init(subkeys[self.num_layers], h)
        z = self.linear_head.apply(w["linear_head"], h).array
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


class FerminetScalarAnsatz:
    def __init__(self, Z, N_up, N_down, config):
        self.Z = Z
        self.N_up = N_up
        self.N_down = N_down
        self.N = N_up + N_down

        self.linear_layer_separate_spins = config.get("linear_layer_separate_spins", False)

        self.num_determinants = config.get("num_determinants", 4)
        self.lmax = config.get("lmax", max(1, max_l_from_Z(Z)))
        assert self.lmax >= 1
        self.hidden_dim = config.get("hidden_dim", 8)

        def get_scalars(x: e3nn.IrrepsArray, print_intermediates=False):
            # x: (..., N, 3) 1o IrrepsArray
            sh_x = e3nn.spherical_harmonics(range(1, self.lmax+1), x, normalize=False, normalization="norm")
            sh_x_norms = e3nn.norm(sh_x)
            if self.N_up == 0 or self.N_down == 0:
                sh_avg = e3nn.mean(sh_x, axis=-2, keepdims=True)
                overlap_sh_avg = e3nn.dot(sh_x, sh_avg, per_irrep=True)
                if print_intermediates:
                    print("sh_x:", sh_x.shape, sh_x.irreps)
                    print(sh_x)
                    print("sh_x_norms:", sh_x_norms.shape, sh_x_norms.irreps)
                    print(sh_x_norms)
                    print("sh_avg:", sh_avg.shape, sh_avg.irreps)
                    print(sh_avg)
                    print("overlap_sh_avg:", overlap_sh_avg.shape, overlap_sh_avg.irreps)
                    print(overlap_sh_avg)
                return e3nn.concatenate([sh_x_norms, overlap_sh_avg])
            else:
                sh_avg_up = e3nn.mean(sh_x[...,:self.N_up,:], axis=-2, keepdims=True)
                overlap_sh_avg_up = e3nn.dot(sh_x, sh_avg_up, per_irrep=True)
                sh_avg_down = e3nn.mean(sh_x[...,self.N_up:,:], axis=-2, keepdims=True)
                overlap_sh_avg_down = e3nn.dot(sh_x, sh_avg_down, per_irrep=True)
                if print_intermediates:
                    print("sh_x:", sh_x.shape, sh_x.irreps)
                    print(sh_x)
                    print("sh_x_norms:", sh_x_norms.shape, sh_x_norms.irreps)
                    print(sh_x_norms)
                    print("sh_avg_up:", sh_avg_up.shape, sh_avg_up.irreps)
                    print(sh_avg_up)
                    print("overlap_sh_avg_up:", overlap_sh_avg_up.shape, overlap_sh_avg_up.irreps)
                    print(overlap_sh_avg_up)
                    print("sh_avg_down:", sh_avg_down.shape, sh_avg_down.irreps)
                    print(sh_avg_down)
                    print("overlap_sh_avg_down:", overlap_sh_avg_down.shape, overlap_sh_avg_down.irreps)
                    print(overlap_sh_avg_down)
                return e3nn.concatenate([sh_x_norms, overlap_sh_avg_up, overlap_sh_avg_down])
        self.get_scalars = get_scalars

        self.num_layers = config.get("num_layers", 4)
        self.layers = []
        for i in range(self.num_layers):
            layer = {
                "averaging": (lambda h: e3nn.mean(h, axis=-2, keepdims=True) if self.N_up == 0 or self.N_down == 0
                    else e3nn.concatenate([e3nn.mean(h[...,:self.N_up,:], axis=-2, keepdims=True),
                    e3nn.mean(h[...,self.N_up:,:], axis=-2, keepdims=True)])),
                "concat": lambda h, avg: e3nn.concatenate([h, e3nn.concatenate([avg] * self.N, axis=-2)]),
                "activation": e3nn.scalar_activation
            }
            if self.linear_layer_separate_spins:
                layer["linear_up"] = e3nn.flax.Linear(self.hidden_dim * e3nn.Irrep(0, 1), biases=True)
                layer["linear_down"] = e3nn.flax.Linear(self.hidden_dim * e3nn.Irrep(0, 1), biases=True)
            else:
                layer["linear"] = e3nn.flax.Linear(self.hidden_dim * e3nn.Irrep(0, 1), biases=True)
            if i > 0:
                layer["residual"] = lambda a, h: a + h
            self.layers.append(layer)

        self.linear_head = e3nn.flax.Linear(irreps_out=self.num_determinants * self.N * e3nn.Irrep(0, 1), biases=True)

        self.envelope = lambda zeta, x: jnp.exp(-zeta * jnp.expand_dims(jnp.linalg.norm(x, axis=-1), axis=-2))

        @jit
        def wavefunction(w, coords):  # coords can be unbatched or batched
            x = e3nn.IrrepsArray("1o", coords.reshape(coords.shape[:-1] + (self.N, 3)))
            h = self.get_scalars(x)
            for i, layer in enumerate(self.layers):
                avg = layer["averaging"](h)
                h_avg = layer["concat"](h, avg)
                if self.linear_layer_separate_spins:
                    z_up = layer["linear_up"].apply(w["linear_hidden_up"][i], h_avg[...,:self.N_up,:])
                    z_down = layer["linear_up"].apply(w["linear_hidden_up"][i], h_avg[...,self.N_up:,:])
                    z = e3nn.concatenate([z_up, z_down], axis=-2)
                else:
                    z = layer["linear"].apply(w["linear_hidden"][i], h_avg)
                a = layer["activation"](z)
                if i == 0:
                    h = a
                else:
                    h = layer["residual"](a, h)
            z_flat = self.linear_head.apply(w["linear_head"], h).array  # (..., N (j), k * N (i))
            num_batch_dims = len(z_flat.shape) - 2
            z = jnp.transpose(z_flat.reshape(z_flat.shape[:-1] + (self.num_determinants, self.N)), axes=tuple(range(num_batch_dims)) + (num_batch_dims+1, num_batch_dims, num_batch_dims+2))
            e = self.envelope(jnp.abs(w["envelope"]) + 0.5, x.array)
            phi = z * jnp.expand_dims(e, axis=-1)
            D_up = jnp.linalg.det(phi[...,:self.N_up,:self.N_up])
            D_down = jnp.linalg.det(phi[...,self.N_up:,self.N_up:])
            return jnp.sum(w["slater_coeffs"] * D_up * D_down, axis=-1)
        self.wavefunction = wavefunction

    def init_weights(self, random_key):
        subkeys = jax.random.split(random_key, num=self.num_layers+3)
        if self.linear_layer_separate_spins:
            w = {"linear_hidden_up": [], "linear_hidden_down": []}
        else:
            w = {"linear_hidden": []}
        coords = jnp.concatenate([jnp.expand_dims(jnp.arange(0.1, 1, 0.2), axis=-1), jnp.zeros((5, 3 * self.N - 1))], axis=-1)
        x = e3nn.IrrepsArray("1o", coords.reshape(coords.shape[:-1] + (self.N, 3)))
        print("x:", x.shape, x.irreps)
        print(x)
        h = self.get_scalars(x, print_intermediates=True)
        print("h:", h.shape, h.irreps)
        print(h)
        for i, layer in enumerate(self.layers):
            avg = layer["averaging"](h)
            print("avg:", avg.shape, avg.irreps)
            print(avg)
            h_avg = layer["concat"](h, avg)
            print("h_avg:", h_avg.shape, h_avg.irreps)
            print(h_avg)
            if self.linear_layer_separate_spins:
                w["linear_hidden_up"].append(layer["linear_up"].init(subkeys[i], h_avg[...,:self.N_up,:]))
                z_up = layer["linear_up"].apply(w["linear_hidden_up"][i], h_avg[...,:self.N_up,:])
                print("z_up:", z_up.shape, z_up.irreps)
                print(z_up)
                w["linear_hidden_down"].append(layer["linear_down"].init(subkeys[i], h_avg[...,self.N_up:,:]))
                z_down = layer["linear_up"].apply(w["linear_hidden_up"][i], h_avg[...,self.N_up:,:])
                print("z_down:", z_down.shape, z_down.irreps)
                print(z_down)
                z = e3nn.concatenate([z_up, z_down], axis=-2)
            else:
                w["linear_hidden"].append(layer["linear"].init(subkeys[i], h_avg))
                z = layer["linear"].apply(w["linear_hidden"][i], h_avg)
            print("z:", z.shape, z.irreps)
            print(z)
            a = layer["activation"](z)
            print("a:", a.shape, a.irreps)
            print(a)
            if i == 0:
                h = a
            else:
                h = layer["residual"](a, h)
            print("h:", h.shape, h.irreps)
            print(h)
        w["linear_head"] = self.linear_head.init(subkeys[self.num_layers], h)
        z_flat = self.linear_head.apply(w["linear_head"], h).array  # (..., N (j), k * N (i))
        print("z_flat:", z_flat.shape)
        print(z_flat)
        num_batch_dims = len(z_flat.shape) - 2
        z = jnp.transpose(z_flat.reshape(z_flat.shape[:-1] + (self.num_determinants, self.N)), axes=tuple(range(num_batch_dims)) + (num_batch_dims+1, num_batch_dims, num_batch_dims+2))
        print("z:", z.shape)
        print(z)
        w["envelope"] = jnp.sqrt(jax.random.chisquare(subkeys[self.num_layers+1], shape=(self.num_determinants, self.N,), df=2)) * 0.5
        print("zeta:", w["envelope"].shape)
        print(w["envelope"])
        e = self.envelope(jnp.abs(w["envelope"]) + 0.5, x.array)
        print("e:", e.shape)
        print(e)
        phi = z * jnp.expand_dims(e, axis=-1)
        print("phi:", phi.shape)
        print(phi)
        D_up = jnp.linalg.det(phi[...,:self.N_up,:self.N_up])
        print("D_up:", D_up.shape)
        print(D_up)
        D_down = jnp.linalg.det(phi[...,self.N_up:,self.N_up:])
        print("D_down:", D_down.shape)
        print(D_down)
        slater_coeffs_unnormalized = jnp.sqrt(jax.random.chisquare(subkeys[self.num_layers+1], shape=(self.num_determinants,), df=2)) * jnp.linspace(1., 0., self.num_determinants, endpoint=False)
        w["slater_coeffs"] = slater_coeffs_unnormalized / jnp.linalg.norm(slater_coeffs_unnormalized)
        psi = jnp.sum(w["slater_coeffs"] * D_up * D_down, axis=-1)
        print("psi:", psi.shape)
        print(psi)
        print("w:")
        print(w)
        return w

    def load_weights(self, ckpt_path):
        with open(ckpt_path, 'rb') as f:
            return pkl.load(f)


class FerminetVanillaAnsatz:
    def __init__(self, Z, N_up, N_down, config):
        self.Z = Z
        self.N_up = N_up
        self.N_down = N_down
        self.N = N_up + N_down

        self.linear_layer_separate_spins = config.get("linear_layer_separate_spins", False)

        self.num_determinants = config.get("num_determinants", 4)
        self.lmax = config.get("lmax", max(1, max_l_from_Z(Z)))
        assert self.lmax >= 1
        self.hidden_dim = config.get("hidden_dim", 8)

        def get_scalars(x, print_intermediates=False):
            # x: (..., N, 3) jnp.ndarray
            norms = jnp.linalg.norm(x, axis=-1, keepdims=True)
            norms = jnp.broadcast_to(norms, x.shape)
            print("norms:", norms.shape)
            print(norms)
            if self.N_up == 0 or self.N_down == 0:
                avg = jnp.mean(x, axis=-2, keepdims=True)
                diffs = x - avg
                norm_diffs = jnp.linalg.norm(diffs, axis=-1, keepdims=True)
                if print_intermediates:
                    print("avg:", avg.shape)
                    print(avg)
                    print("diffs:", diffs.shape)
                    print(diffs)
                    print("norm_diffs:", norm_diffs.shape)
                    print(norm_diffs)
                return jnp.concatenate([x, norms, diffs, norm_diffs], axis=-1)
            else:
                avg_up = jnp.mean(x[...,:self.N_up,:], axis=-2, keepdims=True)
                diffs_up = x - avg_up
                norm_diffs_up = jnp.linalg.norm(diffs_up, axis=-1, keepdims=True)
                avg_down = jnp.mean(x[...,self.N_up:,:], axis=-2, keepdims=True)
                diffs_down = x - avg_down
                norm_diffs_down = jnp.linalg.norm(diffs_down, axis=-1, keepdims=True)
                if print_intermediates:
                    print("avg_up:", avg_up.shape)
                    print(avg_up)
                    print("diffs_up:", diffs_up.shape)
                    print(diffs_up)
                    print("norm_diffs_up:", norm_diffs_up.shape)
                    print(norm_diffs_up)
                    print("avg_down:", avg_down.shape)
                    print(avg_down)
                    print("diffs_down:", diffs_down.shape)
                    print(diffs_down)
                    print("norm_diffs_down:", norm_diffs_down.shape)
                    print(norm_diffs_down)
                return jnp.concatenate([x, norms, diffs_up, norm_diffs_up, diffs_down, norm_diffs_down], axis=-1)
        self.get_scalars = get_scalars

        self.num_layers = config.get("num_layers", 4)
        self.layers = []
        for i in range(self.num_layers):
            layer = {
                "averaging": (lambda h: e3nn.mean(h, axis=-2, keepdims=True) if self.N_up == 0 or self.N_down == 0
                    else e3nn.concatenate([e3nn.mean(h[...,:self.N_up,:], axis=-2, keepdims=True),
                    e3nn.mean(h[...,self.N_up:,:], axis=-2, keepdims=True)])),
                "concat": lambda h, avg: e3nn.concatenate([h, e3nn.concatenate([avg] * self.N, axis=-2)]),
                "activation": e3nn.scalar_activation
            }
            if self.linear_layer_separate_spins:
                layer["linear_up"] = e3nn.flax.Linear(self.hidden_dim * e3nn.Irrep(0, 1), biases=True)
                layer["linear_down"] = e3nn.flax.Linear(self.hidden_dim * e3nn.Irrep(0, 1), biases=True)
            else:
                layer["linear"] = e3nn.flax.Linear(self.hidden_dim * e3nn.Irrep(0, 1), biases=True)
            if i > 0:
                layer["residual"] = lambda a, h: a + h
            self.layers.append(layer)

        self.linear_head = e3nn.flax.Linear(irreps_out=self.num_determinants * self.N * e3nn.Irrep(0, 1), biases=True)

        self.envelope = lambda zeta, x: jnp.exp(-zeta * jnp.expand_dims(jnp.linalg.norm(x, axis=-1), axis=-2))

        @jit
        def wavefunction(w, coords):  # coords can be unbatched or batched
            x = coords.reshape(coords.shape[:-1] + (self.N, 3))
            h = self.get_scalars(x)
            h = e3nn.IrrepsArray(h.shape[-1] * e3nn.Irrep(0, 1), h)
            for i, layer in enumerate(self.layers):
                avg = layer["averaging"](h)
                h_avg = layer["concat"](h, avg)
                if self.linear_layer_separate_spins:
                    z_up = layer["linear_up"].apply(w["linear_hidden_up"][i], h_avg[...,:self.N_up,:])
                    z_down = layer["linear_up"].apply(w["linear_hidden_up"][i], h_avg[...,self.N_up:,:])
                    z = e3nn.concatenate([z_up, z_down], axis=-2)
                else:
                    z = layer["linear"].apply(w["linear_hidden"][i], h_avg)
                a = layer["activation"](z)
                if i == 0:
                    h = a
                else:
                    h = layer["residual"](a, h)
            z_flat = self.linear_head.apply(w["linear_head"], h).array  # (..., N (j), k * N (i))
            num_batch_dims = len(z_flat.shape) - 2
            z = jnp.transpose(z_flat.reshape(z_flat.shape[:-1] + (self.num_determinants, self.N)), axes=tuple(range(num_batch_dims)) + (num_batch_dims+1, num_batch_dims, num_batch_dims+2))
            e = self.envelope(jnp.abs(w["envelope"]), x)
            phi = z * jnp.expand_dims(e, axis=-1)
            D_up = jnp.linalg.det(phi[...,:self.N_up,:self.N_up])
            D_down = jnp.linalg.det(phi[...,self.N_up:,self.N_up:])
            return jnp.sum(w["slater_coeffs"] * D_up * D_down, axis=-1)
        self.wavefunction = wavefunction

    def init_weights(self, random_key):
        subkeys = jax.random.split(random_key, num=self.num_layers+3)
        if self.linear_layer_separate_spins:
            w = {"linear_hidden_up": [], "linear_hidden_down": []}
        else:
            w = {"linear_hidden": []}
        coords = jax.random.normal(random_key, (3 * self.N,))
        x = coords.reshape(coords.shape[:-1] + (self.N, 3))
        h = self.get_scalars(x, print_intermediates=True)
        h = e3nn.IrrepsArray(h.shape[-1] * e3nn.Irrep(0, 1), h)
        print("h:", h.shape, h.irreps)
        print(h)
        for i, layer in enumerate(self.layers):
            avg = layer["averaging"](h)
            print("avg:", avg.shape, avg.irreps)
            print(avg)
            h_avg = layer["concat"](h, avg)
            print("h_avg:", h_avg.shape, h_avg.irreps)
            print(h_avg)
            if self.linear_layer_separate_spins:
                w["linear_hidden_up"].append(layer["linear_up"].init(subkeys[i], h_avg[...,:self.N_up,:]))
                z_up = layer["linear_up"].apply(w["linear_hidden_up"][i], h_avg[...,:self.N_up,:])
                print("z_up:", z_up.shape, z_up.irreps)
                print(z_up)
                w["linear_hidden_down"].append(layer["linear_down"].init(subkeys[i], h_avg[...,self.N_up:,:]))
                z_down = layer["linear_up"].apply(w["linear_hidden_up"][i], h_avg[...,self.N_up:,:])
                print("z_down:", z_down.shape, z_down.irreps)
                print(z_down)
                z = e3nn.concatenate([z_up, z_down], axis=-2)
            else:
                w["linear_hidden"].append(layer["linear"].init(subkeys[i], h_avg))
                z = layer["linear"].apply(w["linear_hidden"][i], h_avg)
            print("z:", z.shape, z.irreps)
            print(z)
            a = layer["activation"](z)
            print("a:", a.shape, a.irreps)
            print(a)
            if i == 0:
                h = a
            else:
                h = layer["residual"](a, h)
            print("h:", h.shape, h.irreps)
            print(h)
        w["linear_head"] = self.linear_head.init(subkeys[self.num_layers], h)
        z_flat = self.linear_head.apply(w["linear_head"], h).array  # (..., N (j), k * N (i))
        print("z_flat:", z_flat.shape)
        print(z_flat)
        num_batch_dims = len(z_flat.shape) - 2
        z = jnp.transpose(z_flat.reshape(z_flat.shape[:-1] + (self.num_determinants, self.N)), axes=tuple(range(num_batch_dims)) + (num_batch_dims+1, num_batch_dims, num_batch_dims+2))
        print("z:", z.shape)
        print(z)
        w["envelope"] = jnp.sqrt(jax.random.chisquare(subkeys[self.num_layers+1], shape=(self.num_determinants, self.N,), df=2) * self.Z)
        e = self.envelope(jnp.abs(w["envelope"]), x)
        print("e:", e.shape)
        print(e)
        phi = z * jnp.expand_dims(e, axis=-1)
        print("phi:", phi.shape)
        print(phi)
        D_up = jnp.linalg.det(phi[...,:self.N_up,:self.N_up])
        print("D_up:", D_up.shape)
        print(D_up)
        D_down = jnp.linalg.det(phi[...,self.N_up:,self.N_up:])
        print("D_down:", D_down.shape)
        print(D_down)
        slater_coeffs_unnormalized = jnp.sqrt(jax.random.chisquare(subkeys[self.num_layers+1], shape=(self.num_determinants,), df=2)) * jnp.linspace(1., 0., self.num_determinants, endpoint=False)
        w["slater_coeffs"] = slater_coeffs_unnormalized / jnp.linalg.norm(slater_coeffs_unnormalized)
        psi = jnp.sum(w["slater_coeffs"] * D_up * D_down, axis=-1)
        print("psi:", psi.shape)
        print(psi)
        return w

    def load_weights(self, ckpt_path):
        with open(ckpt_path, 'rb') as f:
            return pkl.load(f)


ANSATZE = {
        "toy": ToyAnsatz,
        "manual": ManualAnsatz,
        "ferminet": FerminetAnsatz,
        "ferminet_scalar": FerminetScalarAnsatz,
        "ferminet_vanilla": FerminetVanillaAnsatz
}

def Ansatz(Z, N_up, N_down, config):
    return ANSATZE[config.get("type", "ferminet")](Z, N_up, N_down, config)


if __name__ == "__main__":
    for Z in range(1, 120):
        print(Z, max_l_from_Z(Z))
