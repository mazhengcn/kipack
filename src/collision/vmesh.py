import math

import numpy as np

from .spherical_design import get_sphrquadrule


class VMesh(object):
    def __init__(self, config, *args, **kwargs):
        # Get the config
        self.config = config
        # Get dimension
        self._ndim = int(self.config["collision-model"]["dim"])
        print("{} dimensional collision model.".format(self._ndim))
        # Construct the velocity mesh
        self._construct_velocity_mesh()
        # Load quadrature for integration
        self._load_radial_quadrature()
        # Construct the integration mesh on a circle (2D) or sphere (3D)
        if self._ndim == 2:
            self._construct_polar_mesh()
        elif self._ndim == 3:
            self._construct_spherical_mesh()
        else:
            raise ValueError("Dimension must be 2 or 3.")

        self._v_center = None
        self._v_centers = None

    def _load_radial_quadrature(self):
        vm_sect = self.config["velocity-mesh"]
        quad_rule = vm_sect.get("quad-rule", "legendre")
        v_quads = {"legendre": np.polynomial.legendre.leggauss}
        r, wr = v_quads[quad_rule](self._nr)
        self._r = 0.5 * (r + 1) * self._R
        self._wr = 0.5 * self._R * wr

    def _construct_velocity_mesh(self):
        vm_sect = self.config["velocity-mesh"]
        # Define the velocity mesh for 2D and 3D
        self._nv = int(vm_sect["nv"])
        print("Number of velocity cells: {}.".format(self._nv))

        # Number of points on radial direction
        self._nr = int(vm_sect.get("nr", int(self._nv / 2)))

        # Define the physical domain
        self._S = float(vm_sect["s"])
        self._R = 2 * self._S
        self._L = 0.5 * (3.0 + math.sqrt(2)) * self._S
        print("Velocity domain: [{}, {}].".format(-self._L, self._L))

    def _construct_polar_mesh(self):
        circ_quad = self.config["circle-quad-rule"]
        # Number of points on the circle
        self._nphi = int(circ_quad["nphi"])
        self._wphi = 2 * math.pi / self._nphi
        self._phi = np.arange(0, 2 * math.pi, self._wphi)

    def _construct_spherical_mesh(self):
        sphr_sect = self.config["spherical-design-rule"]
        self._ssrule = sphr_sect["ssrule"]
        self._nsphr = int(sphr_sect["nsphr"])
        srule = get_sphrquadrule(
            "symmetric", rule=self._ssrule, npts=self._nsphr
        )
        self._spts = srule.pts
        self._wspts = 4 * math.pi / self._nsphr

    @property
    def v_center(self):
        if self._v_center is None:
            self._v_center = np.empty(self.nv)
            for i in range(self.nv):
                self._v_center[i] = -self.L + (i + 0.5) * self.delta
        return self._v_center

    @property
    def v_centers(self):
        if self._v_centers is None:
            index = np.indices(self.nv_s)
            self._v_centers = [
                self.v_center[index[i, ...]] for i in range(self.ndim)
            ]
        return self._v_centers

    @property
    def ncirc_or_nsphr(self):
        if self.ndim == 2:
            return self._nphi
        elif self.ndim == 3:
            return self._nsphr
        else:
            raise ValueError("Dimension must be 2 or 3.")

    def circ_or_sphr_quad(self):
        if self.ndim == 2:
            sigma = np.stack((np.cos(self._phi), np.sin(self._phi)), axis=-1)
            return sigma, self._wphi
        elif self.ndim == 3:
            return self._spts, self._wspts
        else:
            raise ValueError("Dimension must be 2 or 3.")

    @property
    def ndim(self):
        return self._ndim

    @property
    def nv(self):
        return self._nv

    @property
    def nv_s(self):
        return [self.nv] * self.ndim

    @property
    def L(self):
        return self._L

    @property
    def delta(self):
        return 2 * self.L / self.nv

    @property
    def vsquare(self):
        vsq = 0.0
        for v in self.v_centers:
            vsq += v ** 2
        return vsq

    @property
    def nr(self):
        return self._nr

    def rquad(self):
        return self._r, self._wr

    def get_F(self, f):
        w = self.delta ** (self.ndim)
        vaxis = tuple(-(i + 1) for i in range(self.ndim))
        rho = np.sum(f, axis=vaxis) * w
        m = [np.sum(f * v, axis=vaxis) * w for v in self.v_centers]
        E = 0.5 * np.sum(f * self.vsquare, axis=vaxis) * w

        return rho, m, E

    def get_p(self, f):
        rho, m, E = self.get_F(f)
        u = [m_i / rho for m_i in m]
        usq = 0.0
        for ui in u:
            usq += ui ** 2
        T = (2 * E / rho - usq) / self.ndim

        return rho, u, T
