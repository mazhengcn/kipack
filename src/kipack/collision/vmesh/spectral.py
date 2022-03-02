import math

import numpy as np
from absl import logging
from ml_collections import ConfigDict

from .base import VMesh
from .spherical_design import get_sphrquadrule


class SpectralMesh(VMesh):
    """Vmesh for using spectral method."""

    def __init__(self, config: ConfigDict, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # Load quadrature for integration
        self._load_radial_quadrature()
        # Construct the integration mesh on a circle (2D) or sphere (3D)
        if self._ndim == 2:
            self._build_polar_mesh()
        elif self._ndim == 3:
            self._build_spherical_mesh()
        else:
            raise ValueError("Dimension must be 2 or 3.")

    def _load_radial_quadrature(self):
        quad_rule = self.config.quad_rule
        v_quads = {"legendre": np.polynomial.legendre.leggauss}
        r, wr = v_quads[quad_rule](self._nr)
        self._r = 0.5 * (r + 1) * self._R
        self._wr = 0.5 * self._R * wr

    def _build_velocity_mesh(self):
        # Define the velocity mesh for 2D and 3D
        self._nv = self.config.nv
        logging.info(f"Number of velocity cells: {self._nv}.")

        # Number of points on radial direction
        self._nr = self.config.nr

        # Define the physical domain
        self._S = self.config.s
        self._R = 2 * self._S
        self._L = 0.5 * (3.0 + math.sqrt(2)) * self._S
        logging.info(f"Velocity domain: [{-self._L}, {self._L}].")

    def _build_polar_mesh(self):
        # Number of points on the circle
        self._nphi = self.config.nphi
        self._wphi = 2 * math.pi / self._nphi
        self._phi = np.arange(0, 2 * math.pi, self._wphi)

    def _build_spherical_mesh(self):
        self._ssrule = self.config.ssrule
        self._nsphr = self.config.nsphr
        srule = get_sphrquadrule(
            "symmetric", rule=self._ssrule, npts=self._nsphr
        )
        self._spts = srule.pts
        self._wspts = 4 * math.pi / self._nsphr

    @property
    def center(self):
        if self._center is None:
            self._center = np.empty(self.nv)
            for i in range(self.nv):
                self._center[i] = -self.L + (i + 0.5) * self.delta
        return self._center

    @property
    def nv(self):
        return self._nv

    @property
    def num_nodes(self):
        return [self.nv] * self.num_dim

    @property
    def ncirc_or_nsphr(self):
        if self.num_dim == 2:
            return self._nphi
        elif self.num_dim == 3:
            return self._nsphr
        else:
            raise ValueError("Dimension must be 2 or 3.")

    def circ_or_sphr_quad(self):
        if self.num_dim == 2:
            sigma = np.stack((np.cos(self._phi), np.sin(self._phi)), axis=-1)
            return sigma, self._wphi
        elif self.num_dim == 3:
            return self._spts, self._wspts
        else:
            raise ValueError("Dimension must be 2 or 3.")

    @property
    def lower(self):
        return -self._L

    @property
    def upper(self):
        return self._L

    @property
    def L(self):
        return self._L

    @property
    def nr(self):
        return self._nr

    def rquad(self):
        return self._r, self._wr
