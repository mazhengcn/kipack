import math

import numpy as np

from kipack.collision.spherical_design import get_sphrquadrule


class SpectralMesh(object):
    def __init__(self, config, *args, **kwargs):
        # Get dimension
        self._ndim = config.collision_model.dim
        print("{} dimensional collision model.".format(self._ndim))
        # Get the config
        self.config = config.velocity_mesh
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

        self._center = None
        self._centers = None

    def _load_radial_quadrature(self):
        quad_rule = self.config.quad_rule
        v_quads = {"legendre": np.polynomial.legendre.leggauss}
        r, wr = v_quads[quad_rule](self._nr)
        self._r = 0.5 * (r + 1) * self._R
        self._wr = 0.5 * self._R * wr

    def _construct_velocity_mesh(self):
        # Define the velocity mesh for 2D and 3D
        self._nv = self.config.nv
        print("Number of velocity cells: {}.".format(self._nv))

        # Number of points on radial direction
        self._nr = self.config.nr

        # Define the physical domain
        self._S = self.config.s
        self._R = 2 * self._S
        self._L = 0.5 * (3.0 + math.sqrt(2)) * self._S
        print("Velocity domain: [{}, {}].".format(-self._L, self._L))

    def _construct_polar_mesh(self):
        # Number of points on the circle
        self._nphi = self.config.nphi
        self._wphi = 2 * math.pi / self._nphi
        self._phi = np.arange(0, 2 * math.pi, self._wphi)

    def _construct_spherical_mesh(self):
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
    def centers(self):
        if self._centers is None:
            index = np.indices(self.nvs)
            self._centers = [
                self.center[index[i, ...]] for i in range(self.ndim)
            ]
        return self._centers

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
    def nvs(self):
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
        for v in self.centers:
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
        m = [np.sum(f * v, axis=vaxis) * w for v in self.centers]
        E = 0.5 * np.sum(f * self.vsquare, axis=vaxis) * w

        return [rho, m, E]

    def get_p(self, f):
        rho, m, E = self.get_F(f)
        u = [m_i / rho for m_i in m]
        usq = 0.0
        for ui in u:
            usq += ui ** 2
        T = (2 * E / rho - usq) / self.ndim

        return [rho, u, T]


class CartesianMesh(object):
    def __init__(self, config, *args, **kwargs):
        # Get dimension
        self._ndim = config.collision_model.dim
        print("{} dimensional collision model.".format(self._ndim))
        # Get the config
        self.config = config.velocity_mesh
        self._center = None
        self._centers = None
        self._weights = None
        # Construct the velocity mesh
        self._construct_velocity_mesh()

    def _load_quadrature(self):
        quad_rule = self.config.quad_rule
        v_quads = {"legendre": np.polynomial.legendre.leggauss}
        v, wv = v_quads[quad_rule](self._nv)
        self._center = self.L * v + 0.5 * (self.lower + self.upper)
        self._weights = self.L * wv

    def _construct_velocity_mesh(self):
        # Define the velocity mesh for 2D and 3D
        self._nv = self.config.nv
        print("Number of velocity cells: {}.".format(self._nv))

        # Define the physical domain
        self._lower = self.config.lower
        self._upper = self.config.upper
        print("Velocity domain: [{}, {}].".format(self._lower, self._upper))

        if self.config.quad_rule == "legendre":
            self._load_quadrature()

    @property
    def center(self):
        if self._center is None:
            self._center = np.empty(self.nv)
            for i in range(self.nv):
                self._center[i] = self.lower + (i + 0.5) * self.delta
        return self._center

    @property
    def centers(self):
        if self._centers is None:
            index = np.indices(self.nvs)
            self._centers = [
                self.center[index[i, ...]] for i in range(self.ndim)
            ]
        return self._centers

    @property
    def ndim(self):
        return self._ndim

    @property
    def nv(self):
        return self._nv

    @property
    def nvs(self):
        return [self.nv] * self.ndim

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
    def delta(self):
        return 2 * self.L / self.nv

    @property
    def weights(self):
        if self._weights is not None:
            weights = 1.0
            for _ in range(self.ndim):
                weights = weights * self._weights
            return weights
        else:
            return self.delta ** self.ndim

    @property
    def vsquare(self):
        vsq = 0.0
        for v in self.centers:
            vsq += v ** 2
        return vsq

    def get_F(self, f):
        vaxis = tuple(-(i + 1) for i in range(self.ndim))
        rho = np.sum(f * self.weights, axis=vaxis)
        m = [np.sum(f * v * self.weights, axis=vaxis) for v in self.centers]
        E = 0.5 * np.sum(f * self.vsquare * self.weights, axis=vaxis)

        return [rho, m, E]

    def get_p(self, f):
        rho, m, E = self.get_F(f)
        u = [m_i / rho for m_i in m]
        usq = 0.0
        for ui in u:
            usq += ui ** 2
        T = (2 * E / rho - usq) / self.ndim

        return [rho, u, T]
