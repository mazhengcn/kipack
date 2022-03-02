import math

import numpy as np

from .base import Collision


class RandomBatchCollisionV1(Collision):
    def load_parameters(self):
        # Load collision model (e and gamma)
        self.nv = self.vm.nv
        self.vmin = self.vm.v_center[0]
        self.dv = self.vm.delta
        # vgrid shape: (2, nv, nv)
        self.v = np.asarray(self.vm.v_centers)
        # Get sigma
        self.ncirc = self.vm.ncirc_or_nsphr
        # sigma shape (ncir, 2)
        self.sigma, self.wsigma = self.vm.circ_or_sphr_quad()

    def _random_batch(self):
        # 2 random index arrays with shapes (nv, nv)
        idx_v_ast = np.random.randint(self.nv, size=(2, self.nv, self.nv))
        idx_sigma = np.random.randint(self.ncirc, size=(self.nv, self.nv))
        idx_dict = {"f_ast": idx_v_ast}
        # v_ast shape (2, nv, nv)
        v_ast = self.v[:, idx_v_ast[0], idx_v_ast[1]]
        idx_dict["v_ast"] = v_ast
        rel_v = np.sqrt(np.sum((v_ast - self.v) ** 2, axis=0))
        # rand_sigma shape (2, nv, nv)
        rand_sigma = np.transpose(self.sigma)[:, idx_sigma]
        # Index of v'
        vp = 0.5 * (v_ast + self.v) + 0.5 * rel_v * rand_sigma
        idx_vp = ((vp - self.vmin) // self.dv).astype(int)
        delta_vp = (vp - (self.vmin + idx_vp * self.dv)) / self.dv
        idx_dict["fp"] = [idx_vp, delta_vp]
        # Index of v'_*
        vp_ast = 0.5 * (v_ast + self.v) - 0.5 * rel_v * rand_sigma
        idx_vp_ast = ((vp_ast - self.vmin) // self.dv).astype(int)
        delta_vp_ast = (vp_ast - (self.vmin + idx_vp_ast * self.dv)) / self.dv
        idx_dict["fp_ast"] = [idx_vp_ast, delta_vp_ast]

        return idx_dict

    def _extrap_boundary(self, idx):
        idx, delta = idx
        # Extrapolate boundary as constant
        idx[idx < 0] = 0
        idx[idx > self.nv - 1] = self.nv - 1
        delta[(idx < 0) * (idx > self.nv - 2)] = 0.0
        # Left and right index
        idx_l = idx
        idx_r = (idx_l + 1) % self.nv
        return idx_l, idx_r, delta

    def collide(self, input_f):
        f = input_f
        # Generate random index
        idx = self._random_batch()
        # f_*
        idx_f_ast = idx["f_ast"]
        f_ast = f[..., idx_f_ast[0], idx_f_ast[1]]
        # f'
        idx_fp_l, idx_fp_r, delta_fp = self._extrap_boundary(idx["fp"])
        fp = (
            f[..., idx_fp_l[0], idx_fp_l[1]] * (1 - delta_fp)
            + f[..., idx_fp_r[0], idx_fp_r[1]] * delta_fp
        )
        # f'_*
        idx_fp_ast_l, idx_fp_ast_r, delta_fp_ast = self._extrap_boundary(
            idx["fp_ast"]
        )
        fp_ast = (
            f[..., idx_fp_ast_l[0], idx_fp_ast_l[1]] * (1 - delta_fp_ast)
            + f[..., idx_fp_ast_r[0], idx_fp_ast_r[1]] * delta_fp_ast
        )

        return (
            (fp_ast * fp - f_ast * f)
            / (2 * math.pi)
            * self.dv ** 2
            * self.wsigma
        )

    def perform_precomputation(self):
        pass
