import pickle as pkl
import jax
from jax import jit
import jax.numpy as jnp
import e3nn_jax as e3nn  # import e3nn-jax


class FerminetScalarAnsatzMol:
    def __init__(self, nuclei_coords, N_up, N_down, config):
        self.nuclei_coords = nuclei_coords   # (M, 3) jnp.ndarray
        self.M = len(nuclei_coords)
        self.N_up = N_up
        self.N_down = N_down
        self.N = N_up + N_down

        self.lmax = config.get("lmax", 4)
        assert self.lmax >= 1
        self.hidden_dim = config.get("hidden_dim", 8)

        def get_scalars(x):
            # x: (..., N, 3) jnp.ndarray
            x_rel_array = jnp.expand_dims(x, axis=-2) - self.nuclei_coords  # (..., N, M, 3)
            x_rel = e3nn.IrrepsArray("1o", x_rel_array)
            sh_x_rel = e3nn.spherical_harmonics(range(1, self.lmax+1), x_rel, normalize=False, normalization="norm")  # (..., N, M, (l+1)**2)
            sh_x_rel_norms = e3nn.norm(sh_x_rel)  # (..., N, M, l+1)
            assert sh_x_rel.shape[:-1] == sh_x_rel_norms.shape[:-1] and len(sh_x_rel.shape) == len(sh_x_rel_norms.shape)
            if self.N_up == 0 or self.N_down == 0:
                sh_avg = e3nn.mean(sh_x_rel, axis=-3, keepdims=True)
                overlap_sh_avg = e3nn.dot(sh_x_rel, sh_avg, per_irrep=True)
                scalars = e3nn.concatenate([sh_x_rel_norms, overlap_sh_avg]).array
            else:
                sh_avg_up = e3nn.mean(sh_x_rel[...,:self.N_up,:,:], axis=-3, keepdims=True)
                overlap_sh_avg_up = e3nn.dot(sh_x_rel, sh_avg_up, per_irrep=True)
                sh_avg_down = e3nn.mean(sh_x_rel[...,self.N_up:,:,:], axis=-3, keepdims=True)
                overlap_sh_avg_down = e3nn.dot(sh_x_rel, sh_avg_down, per_irrep=True)
                scalars = e3nn.concatenate([sh_x_rel_norms, overlap_sh_avg_up, overlap_sh_avg_down]).array
            return e3nn.IrrepsArray(self.M * scalars.shape[-1] * e3nn.Irrep(0, 1), scalars.reshape(scalars.shape[:-2] + (-1,)))
        self.get_scalars = get_scalars

        self.num_layers = config.get("num_layers", 4)
        self.layers = []
        for i in range(self.num_layers):
            layer = {
                "averaging": (lambda h: e3nn.mean(h, axis=-2, keepdims=True) if self.N_up == 0 or self.N_down == 0
                    else e3nn.concatenate([e3nn.mean(h[...,:self.N_up,:], axis=-2, keepdims=True),
                    e3nn.mean(h[...,self.N_up:,:], axis=-2, keepdims=True)])),
                "concat": lambda h, avg: e3nn.concatenate([h, e3nn.concatenate([avg] * self.N, axis=-2)]),
                "linear": e3nn.flax.Linear(self.hidden_dim * e3nn.Irrep(0, 1), biases=True),  # separate up and down?
                "activation": e3nn.scalar_activation
            }
            if i > 0:
                layer["residual"] = lambda a, h: a + h
            self.layers.append(layer)

        self.linear_head = e3nn.flax.Linear(irreps_out=self.N * e3nn.Irrep(0, 1), biases=True)
        self.tanh_head = config.get("tanh_head", False)

        def envelope(zeta, beta, x):
            # zeta: (N, M) jnp.ndarray
            # beta: (N, M) jnp.ndarray
            # x: (..., N, 3) jnp.ndarray
            rel_dists = jnp.linalg.norm(jnp.expand_dims(x, axis=-2) - self.nuclei_coords, axis=-1)  # (..., N, M)
            exponents = zeta * jnp.expand_dims(rel_dists, axis=-2)  # (..., N (j), N (i), M)
            envelopes = beta * jnp.exp(-exponents)  # (..., N (j), N (i), M)
            return jnp.sum(envelopes, axis=-1)  # (..., N (j), N (i))
        self.envelope = envelope

        @jit
        def wavefunction(w, coords):  # coords can be unbatched or batched
            x = coords.reshape(coords.shape[:-1] + (self.N, 3))
            h = self.get_scalars(x)
            for i, layer in enumerate(self.layers):
                avg = layer["averaging"](h)
                h_avg = layer["concat"](h, avg)
                z = layer["linear"].apply(w["linear_hidden"][i], h_avg)
                a = layer["activation"](z)
                if i == 0:
                    h = a
                else:
                    h = layer["residual"](a, h)
            z = self.linear_head.apply(w["linear_head"], h).array  # (..., N (j), N (i))
            if self.tanh_head:
                z = jax.nn.tanh(z)
            e = self.envelope(jnp.abs(w["envelope_exponent"]), jnp.abs(w["envelope_coeffs"]) + 0.5, x)  # (..., N (j), N (i))
            phi = z * e  # (..., N (j), N (i))
            D_up = jnp.linalg.det(phi[...,:self.N_up,:self.N_up])
            D_down = jnp.linalg.det(phi[...,self.N_up:,self.N_up:])
            return D_up * D_down
        self.wavefunction = wavefunction

    def init_weights(self, random_key):
        subkeys = jax.random.split(random_key, num=self.num_layers+2)
        w = {"linear_hidden": []}
        coords = jnp.empty(3 * self.N)
        x = coords.reshape(coords.shape[:-1] + (self.N, 3))
        h = self.get_scalars(x)
        print("h:", h.shape, h.irreps)
        for i, layer in enumerate(self.layers):
            avg = layer["averaging"](h)
            print("avg:", avg.shape, avg.irreps)
            h_avg = layer["concat"](h, avg)
            print("h_avg:", h_avg.shape, h_avg.irreps)
            w["linear_hidden"].append(layer["linear"].init(subkeys[i], h_avg))
            z = layer["linear"].apply(w["linear_hidden"][i], h_avg)
            print("z:", z.shape, z.irreps)
            a = layer["activation"](z)
            print("a:", a.shape, a.irreps)
            if i == 0:
                h = a
            else:
                h = layer["residual"](a, h)
            print("h:", h.shape, h.irreps)
        w["linear_head"] = self.linear_head.init(subkeys[self.num_layers], h)
        z = self.linear_head.apply(w["linear_head"], h).array
        if self.tanh_head:
            z = jax.nn.tanh(z)
        print("z:", z.shape)
        w["envelope_exponent"] = jnp.sqrt(jax.random.chisquare(subkeys[self.num_layers+1], shape=(self.N, self.M), df=2)) * 0.5
        w["envelope_coeffs"] = jnp.sqrt(jax.random.chisquare(subkeys[self.num_layers+2], shape=(self.N, self.M), df=2)) * 0.5
        e = self.envelope(jnp.abs(w["envelope_exponent"]) + 0.5, jnp.abs(w["envelope_coeffs"]) + 0.5, x)  # (..., N (j), N (i))
        print("e:", e.shape)
        phi = z * e
        print("phi:", phi.shape)
        D_up = jnp.linalg.det(phi[...,:self.N_up,:self.N_up])
        print("D_up:", D_up.shape)
        D_down = jnp.linalg.det(phi[...,self.N_up:,self.N_up:])
        print("D_down:", D_down.shape)
        psi = D_up * D_down
        print("psi:", psi)
        return w

    def load_weights(self, ckpt_path):
        with open(ckpt_path, 'rb') as f:
            return pkl.load(f)


ANSATZE = {
        "ferminet_scalar_mol": FerminetScalarAnsatzMol
}

def Ansatz(Z, N_up, N_down, config):
    return ANSATZE[config.get("type", "ferminet_scalar_mol")](Z, N_up, N_down, config)
