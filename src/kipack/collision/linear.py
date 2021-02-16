import cupy as cp
import numpy as np
from kipack.collision.base import BaseCollision


class LinearCollision(BaseCollision):
    def load_parameters(self):
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

    def collide(self, input_f):
        xp = cp.get_array_module(input_f)
        f = input_f
        vaxis = tuple(-(i + 1) for i in range(self.num_dim))

        return xp.sum(f * self.weights, axis=vaxis, keepdims=True) - f

    def perform_precomputation(self):
        pass
