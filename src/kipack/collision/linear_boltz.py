# import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np

from .base import Collision


class LinearBotlzmannCollision(Collision):
    def __init__(
        self, config, velocity_mesh, sigma=None, heat_bath=None, device="cpu"
    ):
        self.sigma = sigma
        super().__init__(config, velocity_mesh, heat_bath, device)

        self._collide = jax.jit(self._collide)

    def collide(self, input_f):
        return self._collide(input_f)

    def load_parameters(self):
        self.nv = self.vm.nv

    def _build_cpu(self, input_shape):
        self._input_shape = input_shape
        self._built = True

    def _build_gpu(self, input_shape):
        self._input_shape = input_shape
        self._built = True

    def _collide(self, input_f: np.ndarray | jax.Array):
        f = input_f
        if self.num_dim == 1:
            gain = self.maxwellian_mat * jnp.dot(
                f, self.exp_mat * self.sigma_mat * self.weights
            )
        else:
            raise ValueError("Only dimension 1 is implemented.")

        loss = self.col_freq * f

        return gain - loss

    def perform_precomputation(self):
        v = self.vm.center
        vaxis = tuple(-(i + 1) for i in range(self.num_dim))

        self.weights = jnp.asarray(self.vm.weights)
        self.sigma_mat = jnp.asarray(self.sigma(v, v[:, None]))
        self.col_freq = jnp.sum(self.sigma_mat * self.weights, axis=vaxis)
        self.maxwellian_mat = jnp.exp(-(v**2))
        self.exp_mat = jnp.exp(v**2)
        self.index_array = jnp.arange(self.nv)

        print("Collision model precomputation finished!")


class RandomBatchLinearBoltzmannCollision(LinearBotlzmannCollision):
    def __init__(
        self,
        config,
        velocity_mesh,
        seed=0,
        sigma=None,
        heat_bath=None,
        device="cpu",
    ):
        super().__init__(config, velocity_mesh, sigma, heat_bath, device)
        self.key = jax.random.PRNGKey(seed)

    def collide(self, input_f):
        idx = self._random_batch()
        coll = self._collide(input_f, idx)
        return coll

    def _random_batch(self):
        self.key, subkey = jax.random.split(self.key)
        idx = jax.random.randint(subkey, (self.nv,), 0, self.nv)
        return idx

    def _collide(self, input_f, idx):
        f = input_f
        if self.num_dim == 1:
            f_batch = f[..., idx]
            weights_batch = self.nv * self.weights[idx]
            sigma_batch = self.sigma_mat[self.index_array, idx]
            exp_batch = self.exp_mat[idx]
        elif self.num_dim == 2:
            f_batch = f[..., idx, idx]
            weights_batch = self.nv * self.weights[idx, idx]

        col = (
            self.maxwellian_mat
            * exp_batch
            * sigma_batch
            * f_batch
            * weights_batch
            - self.col_freq * f
        )
        return col


class SymmetricRBMLinearBoltzmannCollision(
    RandomBatchLinearBoltzmannCollision
):
    def collide(self, input_f):
        nvrange = self._random_batch()
        coll = self._collide(input_f, nvrange)
        return coll

    def _random_batch(self):
        self.key, subkey = jax.random.split(self.key)
        nvrange = jax.random.permutation(subkey, self.nv)
        # idx = jnp.empty(self.nv, dtype=jnp.int32)
        # idx = idx.at[nvrange[: self.nv // 2]].set(nvrange[self.nv // 2 :])
        # idx = idx.at[nvrange[self.nv // 2 :]].set(nvrange[: self.nv // 2])
        # idx[nvrange[: self.nv // 2]] = nvrange[self.nv // 2 :]
        # idx[nvrange[self.nv // 2 :]] = nvrange[: self.nv // 2]
        return nvrange

    def _collide(self, input_f, nvrange):
        idx = jnp.empty(self.nv, dtype=jnp.int32)
        idx = idx.at[nvrange[: self.nv // 2]].set(nvrange[self.nv // 2 :])
        idx = idx.at[nvrange[self.nv // 2 :]].set(nvrange[: self.nv // 2])
        return super()._collide(input_f, idx)
