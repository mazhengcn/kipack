import math

import cupy as cp
import numpy as np
from kipack.collision.base import BaseCollision


class RandomBatchCollisionParticle(BaseCollision):
    def load_parameters(self):
        self.eps = None
        # Load collision model (e and gamma)
        self.nv = self.vm.nv
        self.vmin = self.vm.center[0]
        self.dv = self.vm.delta
        # vgrid shape: (2, nv, nv)
        self._cpu_v = np.asarray(self.vm.centers)
        # Get sigma
        self.ncirc = self.vm.ncirc_or_nsphr
        # sigma shape (ncir, 2)
        self._cpu_sigma, self.wsigma = self.vm.circ_or_sphr_quad()

        self._built_cpu = True

    def _build_cpu(self, input_shape):
        self._input_shape = input_shape

    def _build_gpu(self, input_shape):
        self._gpu_v = cp.asarray(self._cpu_v)
        self._gpu_sigma = cp.asarray(self._cpu_sigma)
        self._input_shape = input_shape
        self._built_gpu = True

    def _set_to_gpu(self):
        self.v = self._gpu_v
        self.sigma = self._gpu_sigma

    def _set_to_cpu(self):
        self.v = self._cpu_v
        self.sigma = self._cpu_sigma

    def _random_batch(self, xp):
        # 2 random index arrays with shapes (nv, nv)
        idx_v_ast = xp.random.randint(self.nv, size=(2, self.nv, self.nv))
        idx_sigma = xp.random.randint(self.ncirc, size=(self.nv, self.nv))
        batch_dict = {"f_ast": idx_v_ast}
        # v_ast shape (2, nv, nv)
        v_ast = self.v[:, idx_v_ast[0], idx_v_ast[1]]
        rel_v = xp.sqrt(xp.sum((v_ast - self.v) ** 2, axis=0))
        # rand_sigma shape (2, nv, nv)
        rand_sigma = xp.transpose(self.sigma)[:, idx_sigma]
        # v'
        vp = 0.5 * (v_ast + self.v) + 0.5 * rel_v * rand_sigma
        batch_dict["vp"] = vp
        # v'_*
        vp_ast = 0.5 * (v_ast + self.v) - 0.5 * rel_v * rand_sigma
        batch_dict["vp_ast"] = vp_ast

        return batch_dict

    def _particle_f(self, f, v, eps, xp):
        v = v[..., None, None]
        vgrid = self.v[:, None, None, ...]
        kernel = xp.exp(-np.sum((v - vgrid) ** 2, axis=0) / (2 * eps)) / (
            2 * math.pi * eps
        )

        return xp.sum(f[..., None, None, :, :] * kernel, axis=(-1, -2))

    def collide(self, input_f):
        xp = cp.get_array_module(input_f)
        f = input_f
        # Generate random index
        batch = self._random_batch(xp)
        # f_*
        idx_f_ast = batch["f_ast"]
        f_ast = f[..., idx_f_ast[0], idx_f_ast[1]]
        # f'
        vp = batch["vp"]
        fp = self._particle_f(f * self.dv ** 2, vp, self.eps, xp)
        # f'_*
        vp_ast = batch["vp_ast"]
        fp_ast = self._particle_f(f * self.dv ** 2, vp_ast, self.eps, xp)

        return 4 * self.vm.L ** 2 * (fp_ast * fp - f_ast * f)

    def perform_precomputation(self):
        pass
