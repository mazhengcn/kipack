import dataclasses

import jax
import jax.numpy as jnp
from absl import logging

from .base import Array, Collision


@dataclasses.dataclass
class LinearBotlzmannCollision(Collision):
    sigma: float | Array = 1.0

    def __post_init__(self):
        super().__post_init__()
        # Load model parameters
        if self.num_dim != 1:
            raise ValueError("Only dimension 1 is implemented.")

        self.nv = self.vm.nv

    def setup(self, input_shape):
        vaxis = tuple(-(i + 1) for i in range(self.num_dim))
        v = self.vm.center

        self.weights = jnp.asarray(self.vm.weights)
        self.sigma_mat = jnp.asarray(self.sigma(v, v[:, None]))
        self.col_freq = jnp.sum(self.sigma_mat * self.weights, axis=vaxis)
        self.maxwellian_mat = jnp.exp(-(v**2))
        self.exp_mat = jnp.exp(v**2)
        self.index_array = jnp.arange(self.nv)

        logging.info("Collision model precomputation finished!")

    def collide(self, f: Array, rng: jax.Array | None) -> Array:
        gain = self.maxwellian_mat * jnp.dot(
            f, self.exp_mat * self.sigma_mat * self.weights
        )
        loss = self.col_freq * f

        return gain - loss


@dataclasses.dataclass
class RandomBatchLinearBoltzmannCollision(LinearBotlzmannCollision):
    def _random_batch(self, rng):
        return jax.random.randint(rng, (self.nv,), 0, self.nv)

    def collide(self, f: Array, rng: jax.Array) -> Array:
        idx = self._random_batch(rng)

        if self.num_dim == 1:
            f_batch = f[..., idx]
            weights_batch = self.nv * self.weights[idx]
            sigma_batch = self.sigma_mat[self.index_array, idx]
            exp_batch = self.exp_mat[idx]

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
    def _random_batch(self, rng):
        nvrange = jax.random.permutation(rng, self.nv)
        idx = jnp.empty(self.nv, dtype=jnp.int32)
        idx = idx.at[nvrange[: self.nv // 2]].set(nvrange[self.nv // 2 :])
        idx = idx.at[nvrange[self.nv // 2 :]].set(nvrange[: self.nv // 2])
        return idx
