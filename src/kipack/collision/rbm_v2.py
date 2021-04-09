import math

import numpy as np
import cupy as cp

from kipack.collision.base import BaseCollision


def collision_kernel(idx_k, idx_l, xp):
    k_dot_l = idx_k[0] * idx_l[0] + idx_k[1] * idx_l[1]
    return (k_dot_l == 0) * 2 * xp.sum(idx_k ** 2, axis=0) + xp.sum(
        idx_l ** 2, axis=0
    ) / 4 * math.pi


class RandomBatchCollisionV2(BaseCollision):
    def load_parameters(self):
        self.eps = None
        # Load collision model (e and gamma)
        self.nv = self.vm.nv
        self.vmin = self.vm.center[0]
        self.dv = self.vm.delta
        # Create index
        idx_x = np.arange(-int(self.nv / 2), int(self.nv / 2))
        idx_i = np.meshgrid(idx_x, idx_x)
        self._cpu_idx_i = np.asarray(idx_i)
        # N_tilde
        self.n_tilde = int(self.nv / 2)
        # vgrid shape: (2, nv, nv)
        self._cpu_v = np.asarray(self.vm.centers)

        self._built_cpu = True

    def _build_cpu(self, input_shape):
        self._input_shape = input_shape

    def _build_gpu(self, input_shape):
        self._gpu_v = cp.asarray(self._cpu_v)
        self._gpu_idx_i = cp.asarray(self._cpu_idx_i)
        self._input_shape = input_shape
        self._built_gpu = True

    def _set_to_gpu(self):
        self.v = self._gpu_v
        self.idx_i = self._gpu_idx_i

    def _set_to_cpu(self):
        self.v = self._cpu_v
        self.idx_i = self._cpu_idx_i

    def _random_batch(self, xp):
        # 2 random index arrays with shapes (nv, nv)
        idx_k = xp.random.randint(
            -self.n_tilde, self.n_tilde, size=(2, self.nv, self.nv)
        )
        idx_l = xp.random.randint(
            -self.n_tilde, self.n_tilde, size=(2, self.nv, self.nv)
        )

        return idx_k, idx_l

    def collide(self, input_f):
        xp = cp.get_array_module(input_f)
        f = input_f
        # Generate random index
        idx_k, idx_l = self._random_batch(xp)

        ik = (self.idx_i + idx_k) % self.nv
        kl = (self.idx_i + idx_l) % self.nv
        ikl = (self.idx_i + idx_k + idx_l) % self.nv

        fp_ast = f[..., ik[0], ik[1]]
        fp = f[..., kl[0], kl[1]]
        f_ast = f[..., ikl[0], ikl[1]]

        return (
            collision_kernel(idx_k, idx_l, xp)
            * (fp_ast * fp - f_ast * f)
            * self.dv ** 2
        )

    def perform_precomputation(self):
        pass
