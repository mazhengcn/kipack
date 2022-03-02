import cupy as cp
import numpy as np

from .base import Collision


class LinearBotlzmannCollision(Collision):
    def load_parameters(self):
        self.nv = self.vm.nv

    def _build_cpu(self, input_shape):
        self._cpu_weights = self._weights
        self._cpu_sigma_mat = self._sigma_mat
        self._cpu_col_freq = self._col_freq
        self._cpu_maxwellian_mat = self._maxwellian_mat
        self._cpu_exp_mat = self._exp_mat
        self._cpu_index_array = self._index_array

        self._input_shape = input_shape
        self._built_cpu = True

    def _build_gpu(self, input_shape):
        self._gpu_weights = cp.asarray(self._weights)
        self._gpu_sigma_mat = cp.asarray(self._sigma_mat)
        self._gpu_col_freq = cp.asarray(self._col_freq)
        self._gpu_maxwellian_mat = cp.asarray(self._maxwellian_mat)
        self._gpu_exp_mat = cp.asarray(self._exp_mat)
        self._gpu_index_array = cp.asarray(self._index_array)

        self._input_shape = input_shape
        self._built_gpu = True

    def _set_to_gpu(self):
        self.weights = self._gpu_weights
        self.sigma_mat = self._gpu_sigma_mat
        self.col_freq = self._gpu_col_freq
        self.maxwellian_mat = self._gpu_maxwellian_mat
        self.exp_mat = self._gpu_exp_mat
        self.index_array = self._gpu_index_array

    def _set_to_cpu(self):
        self.weights = self._cpu_weights
        self.sigma_mat = self._cpu_sigma_mat
        self.col_freq = self._cpu_col_freq
        self.maxwellian_mat = self._cpu_maxwellian_mat
        self.exp_mat = self._cpu_exp_mat
        self.index_array = self._cpu_index_array

    def collide(self, input_f):
        xp = cp.get_array_module(input_f)
        f = input_f
        # vaxis = tuple(-(i + 1) for i in range(self.num_dim))
        if self.num_dim == 1:
            gain = self.maxwellian_mat * xp.dot(
                f, self.exp_mat * self.sigma_mat * self.weights
            )
        else:
            raise ValueError("Only dimension 1 is implemented.")

        loss = self.col_freq * f

        return gain - loss

    def perform_precomputation(self):
        v = self.vm.center
        vaxis = tuple(-(i + 1) for i in range(self.num_dim))

        self._weights = np.asarray(self.vm.weights)
        self._sigma_mat = self.sigma(v, v[:, None])
        self._col_freq = np.sum(self._sigma_mat * self._weights, axis=vaxis)
        self._maxwellian_mat = np.exp(-(v ** 2))
        self._exp_mat = np.exp(v ** 2)
        self._index_array = np.arange(self.nv)

        print("Collision model precomputation finished!")


class RandomBatchLinearBoltzmannCollision(LinearBotlzmannCollision):
    def _random_batch(self, xp):
        idx = xp.random.randint(self.nv, size=(self.nv,) * self.num_dim)
        return idx

    def collide(self, input_f):
        xp = cp.get_array_module(input_f)
        f = input_f
        idx = self._random_batch(xp)
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


class SymmetricRBMLinearCollision(RandomBatchLinearBoltzmannCollision):
    def _random_batch(self, xp):
        nvrange = xp.random.permutation(self.nv)
        idx = xp.empty(self.nv, dtype=int)
        idx[nvrange[: self.nv // 2]] = nvrange[self.nv // 2 :]
        idx[nvrange[self.nv // 2 :]] = nvrange[: self.nv // 2]
        return idx
