import abc
import dataclasses

import jax
import numpy as np
from ml_collections import ConfigDict

from .vmesh.base import VMesh

Array = jax.Array | np.ndarray


@dataclasses.dataclass
class Collision(abc.ABC):
    config: ConfigDict
    vm: VMesh
    name: str = "collision"

    @abc.abstractmethod
    def setup(self, *args):
        pass

    @abc.abstractmethod
    def collide(self, input_f):
        pass

    def __post_init__(self):
        # Load dimension
        self.num_dim = self.config.collision_model.dim
        assert (
            self.num_dim == self.vm.num_dim
        ), "'spectral_mesh' has different \
            velocity dimensions from collision model! This may be caused by \
            using different config files. Please check the consistency."

        self._is_setup = False

    def __call__(self, f: Array) -> Array:
        if not self._is_setup:
            self.setup(f.shape)

        return self.collide(f)

    # get primitive macroscopic quantities [rho, u, T]
    def get_p(self, input_f: Array):
        return self.vm.get_p(input_f)

    # get conserved macroscopic quantities [rho, m, E]
    def get_F(self, input_f: Array):
        return self.vm.get_F(input_f)
