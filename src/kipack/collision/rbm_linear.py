import cupy as cp
import numpy as np
from kipack.collision.base import BaseCollision


class RandomBatchLinearCollision(BaseCollision):
    def load_parameters(self):
        self.nv = self.vm.nv
        # print(self.nv)
        self._cpu_weights = np.asarray(self.vm.weights)
        self._built_cpu = True

    def _build_cpu(self, input_shape):
        self._input_shape = input_shape

    def _build_gpu(self, input_shape):
        self._gpu_weights = cp.asarray(self._cpu_weights)
        self._input_shape = input_shape
        self._built_gpu = True

    def _set_to_gpu(self):
        self.weights = self._gpu_weights

    def _set_to_cpu(self):
        self.weights = self._cpu_weights

    def _random_batch(self, xp):
        idx = xp.random.randint(self.nv, size=self.nv)
        return idx

    def collide(self, input_f):
        xp = cp.get_array_module(input_f)
        f = input_f
        idx = self._random_batch(xp)
        # print(idx)
        return (
            self.nv * 0.5 ** (self.num_dim) * f[..., idx] * self.weights[idx]
            - f
        )

    def perform_precomputation(self):
        pass


class SymmetricRBMLinearCollision(RandomBatchLinearCollision):
    def _random_batch(self, xp):
        nvrange = xp.random.permutation(self.nv)
        idx = xp.empty(self.nv, dtype=int)
        idx[nvrange[: self.nv // 2]] = nvrange[self.nv // 2 :]
        idx[nvrange[self.nv // 2 :]] = nvrange[: self.nv // 2]
        return idx

    def collide(self, input_f):
        xp = cp.get_array_module(input_f)
        f = input_f
        idx = self._random_batch(xp)
        return (
            self.nv * 0.5 ** (self.num_dim) * f[..., idx] * self.weights[idx]
            - f
        )
