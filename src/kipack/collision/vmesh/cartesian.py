import numpy as np
from absl import logging

from .base import VMesh


class CartesianMesh(VMesh):
    def _load_quadrature(self):
        quad_rule = self.config.quad_rule
        v_quads = {
            "legendre": np.polynomial.legendre.leggauss,
            "hermite": np.polynomial.hermite.hermgauss,
        }
        v, wv = v_quads[quad_rule](self._nv)
        if quad_rule in ["legendre"]:
            self._center = self.L * v + 0.5 * (self.lower + self.upper)
            self._weights = self.L * wv
        else:
            self._center = v
            self._weights = wv

    def _build_velocity_mesh(self):
        # Define the velocity mesh for 2D and 3D
        self._nv = self.config.nv
        logging.info(f"Number of velocity cells: {self._nv}.")

        # Define the physical domain
        self._lower = self.config.lower
        self._upper = self.config.upper
        logging.info(f"Velocity domain: [{self._lower}, {self._upper}].")

        if self.config.quad_rule != "uniform":
            self._load_quadrature()

    @property
    def center(self):
        if self._center is None:
            self._center = np.empty(self.nv)
            for i in range(self.nv):
                self._center[i] = self.lower + (i + 0.5) * self.delta
        return self._center

    @property
    def weights(self):
        if self._weights is not None:
            weights = 1.0
            for _ in range(self.num_dim):
                weights = weights * self._weights / (2 * self.L)
            return weights
        else:
            return (self.delta / (2 * self.L)) ** self.num_dim * np.ones(self.num_nodes)

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    # half of the domain length
    @property
    def L(self):
        return (self.upper - self.lower) / 2

    @property
    def nv(self):
        return self._nv

    @property
    def num_nodes(self):
        return [self.nv] * self.num_dim
