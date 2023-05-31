import dataclasses

import jax
import jax.numpy as jnp

from .base import Array, Collision


@dataclasses.dataclass
class LinearCollision(Collision):
    """A simple linear collision, e.g., linear transport collision."""

    def __post_init__(self):
        super().__post_init__()

        self.nv = self.vm.nv

    def setup(self, input_shape):
        self.weights = jnp.asarray(self.vm.weights)

    def collide(self, f: Array, rng: Array | None = None) -> Array:
        vaxis = tuple(-(i + 1) for i in range(self.num_dim))
        out = jnp.sum(f * self.weights, axis=vaxis, keepdims=True) - f
        return out


@dataclasses.dataclass
class RandomBatchLinearCollision(LinearCollision):
    seed: int = 0

    def __post_init__(self):
        super().__post_init__()

        if self.num_dim == 1:
            self.collide = self._collide_1d
        elif self.num_dim == 2:
            self.collide = self._collide_2d
        else:
            raise ValueError("Only dimension 1 and 2 are implemented.")

    def _random_batch(self, rng: Array):
        rng, sub_rng = jax.random.split(rng)
        idx = jax.random.randint(sub_rng, (self.nv,) * self.num_dim, 0, self.nv)
        return idx, rng

    def _collide_1d(self, f: Array, rng: Array):
        idx, rng = self._random_batch(rng)
        f_batch = f[..., idx]
        weights_batch = self.nv * self.weights[idx]
        out = f_batch * weights_batch - f
        return out, rng

    def _collide_2d(self, f: Array, rng: Array):
        idx, rng = self._random_batch(rng)
        f_batch = f[..., idx, idx]
        weights_batch = self.nv * self.weights[idx, idx]
        out = f_batch * weights_batch - f
        return out, rng


class SymmetricRBMLinearCollision(RandomBatchLinearCollision):
    def _random_batch(self, rng: Array):
        rng, sub_rng = jax.random.split(rng)
        nvrange = jax.random.permutation(sub_rng, self.nv)
        idx = jnp.empty(self.nv, dtype=int)
        idx[nvrange[: self.nv // 2]] = nvrange[self.nv // 2 :]
        idx[nvrange[self.nv // 2 :]] = nvrange[: self.nv // 2]
        return idx, rng
