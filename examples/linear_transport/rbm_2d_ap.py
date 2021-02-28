import copy
import math

import numpy as np
from examples.utils import Progbar
from kipack import collision, pykinetic
from kipack.pykinetic.boltzmann.solver import BoltzmannSolver2D


def phi(eps):
    return np.minimum(1.0, 1.0 / eps)


def maxwellian_vec_init(v, u, T, rho):
    return (
        rho[:, None]
        / np.sqrt(2 * math.pi * T)
        * np.exp(-((v - u) ** 2) / (2 * T))
    )


def qinit(state, vmesh, kn, init_func=None):
    x = state.grid.x.centers
    rho = 1.0 + 2.0 * np.cos(4 * math.pi * x)
    v = vmesh.center

    pos_v = init_func(v, 0.0, 1.0, rho)
    # print(pos_v)
    min_v = init_func(-v, 0.0, 1.0, rho)
    state.q[0, :] = 0.5 * (pos_v + min_v)
    state.q[1, :] = 0.5 / kn * (pos_v - min_v)


def compute_rho(state, vmesh):
    vaxis = tuple(-(i + 1) for i in range(vmesh.num_dim))
    return np.sum(state.q[0, ...] * vmesh.weights, axis=vaxis)


class APNeutronTransportSolver2D(BoltzmannSolver2D):
    def __init__(
        self,
        riemann_solver,
        collision_operator,
        kn,
        sigma_s,
        sigma_a,
        Q,
        **kwargs
    ):
        self.sigma_s, self.sigma_a, self.Q = map(
            self._convert_params, [sigma_s, sigma_a, Q]
        )

        super().__init__(
            riemann_solver=riemann_solver,
            collision_operator=collision_operator,
            kn=kn,
            **kwargs,
        )

    def step_collision(self, state):
        dt_kn2 = self.dt / self.kn ** 2

        # Update r
        rho = (
            self.coll[0](state.q[::2], heat_bath=self.tau, device=self.device)
            + state.q[::2]
        )
        rho = 0.5 * (rho[0] + rho[2])
        state.q[::2] = (dt_kn2 * self.sigma_s * rho + state.q[::2]) / (
            1.0 + dt_kn2 * self.sigma_s
        )

        vx, vy = state.problem_data["v"]
        # Compute vx * dr / dx
        num_ghost = self.num_ghost
        self._apply_bcs(state)
        rbc = self.qbc[::2]
        vxdr_dx = vx * (
            rbc[:, num_ghost + 1 : -num_ghost + 1]
            - rbc[:, num_ghost - 1 : -num_ghost - 1]
        )
        vxdr_dx /= 2 * state.grid.delta[0]
        # Compute vy * dr / dy
        vydr_dy = vy * (
            rbc[:, :, num_ghost + 1 : -num_ghost + 1]
            - rbc[:, :, num_ghost - 1 : -num_ghost - 1]
        )
        vydr_dy /= 2 * state.grid.delta[1]
        vydr_dy[0] = -vydr_dy[0]

        # Update j
        phi = state.problem_data["phi"]
        state.q[1::2] = state.q[1::2] - dt_kn2 * (1.0 - self.kn ** 2 * phi) * (
            vxdr_dx + vydr_dy
        )
        state.q[1::2] /= 1.0 + dt_kn2 * self.sigma_s

    def dq(self, state):
        self.step_collision(state)
        deltaq = self.dq_hyperbolic(state)
        deltaq -= self.sigma_a * state.q * self.dt
        deltaq[::2] += self.Q * self.dt
        return deltaq


def run(
    kn=lambda x, y: 1.0,
    sigma_s=lambda x, y: 1.0,
    sigma_a=lambda x, y: 0.0,
    Q=lambda x, y: 0.0,
    xmin=[0.0, 0.0],
    xmax=[1.0, 1.0],
    nx=40,
    dt=0.01,
    nt=1000,
    coll="linear",
    scheme="Euler",
    BC="periodic",
    f_l=lambda vx, vy: 0.0,
    f_b=lambda vx, vy: 0.0,
    f_r=lambda vx, vy: 0.0,
    f_t=lambda vx, vy: 0.0,
    init_func=lambda vmesh, u, T, rho: 0.0,
):
    # Load config
    config = collision.utils.CollisionConfig.from_json(
        "./examples/linear_transport/configs/" + "parity" + ".json"
    )

    # Collision
    vmesh = collision.CartesianMesh(config)
    # print(vmesh.centers[0], vmesh.weights)
    if coll == "linear":
        coll_op = collision.LinearCollision(config, vmesh)
    elif coll == "rbm":
        coll_op = collision.RandomBatchLinearCollision(config, vmesh)
    elif coll == "rbm_symm":
        coll_op = collision.SymmetricRBMLinearCollision(config, vmesh)
    else:
        raise NotImplementedError(
            "Collision method {} is not implemented.".format(coll)
        )

    # x domian
    x = pykinetic.Dimension(xmin[0], xmax[0], nx, name="x")
    y = pykinetic.Dimension(xmin[1], xmax[1], nx, name="y")
    # print(x.centers_with_ghost(2))
    domain = pykinetic.Domain([x, y])

    # Riemann solver
    rp = pykinetic.riemann.parity_1D
    solver = APNeutronTransportSolver2D(
        rp,
        [coll_op],
        kn=kn(x.centers),
        sigma_s=sigma_s(x.centers),
        sigma_a=sigma_a(x.centers),
        Q=Q(x.centers),
    )
    # print(solver.kn)
    solver.order = 1
    # solver.lim_type = -1
    solver.time_integrator = scheme
    solver.dt = dt
    print("dt is {}".format(solver.dt))

    sigma = sigma_s(x.centers) + kn(x.centers) ** 2 * sigma_a(x.centers)
    sigma_l, sigma_r = None, None
    if isinstance(sigma, np.ndarray):
        sigma_l, sigma_r = sigma[0], sigma[-1]
    else:
        sigma_l = sigma_r = sigma
    kn_l, kn_r = None, None
    if isinstance(solver.kn, np.ndarray):
        kn_l, kn_r = solver.kn[0], solver.kn[-1]
    else:
        kn_l = kn_r = solver.kn

    # Boundary conditions
    def dirichlet_lower_BC(state, dim, t, qbc, auxbc, num_ghost):
        v = state.problem_data["v"][0] / sigma_l
        kn_dx = kn_l / state.grid.delta[0]
        for i in range(num_ghost):
            qbc[0, i, :] = (
                f_l(v) - (0.5 - kn_dx * v) * qbc[0, num_ghost, :]
            ) / (0.5 + kn_dx * v)
        for i in range(num_ghost):
            qbc[1, i, :] = (
                2 * f_l(v) - (qbc[0, num_ghost - 1, :] + qbc[0, num_ghost, :])
            ) / kn_l - qbc[1, num_ghost, :]

    def dirichlet_upper_BC(state, dim, t, qbc, auxbc, num_ghost):
        v = state.problem_data["v"][0] / sigma_r
        kn_dx = kn_r / state.grid.delta[0]
        for i in range(num_ghost):
            qbc[0, -i - 1, :] = (
                f_r(v) - (0.5 - kn_dx * v) * qbc[0, -num_ghost - 1, :]
            ) / (0.5 + kn_dx * v)
        for i in range(num_ghost):
            qbc[1, -i - 1, :] = (
                (qbc[0, -num_ghost, :] + qbc[0, -num_ghost - 1, :])
                - 2 * f_r(v)
            ) / kn_r - qbc[1, -num_ghost - 1, :]

    if BC == "periodic":
        solver.bc_lower[0] = pykinetic.BC.periodic
        solver.bc_upper[0] = pykinetic.BC.periodic
    elif BC == "dirichlet":
        solver.bc_lower[0] = pykinetic.BC.custom
        solver.bc_upper[0] = pykinetic.BC.custom
        solver.user_bc_lower = dirichlet_lower_BC
        solver.user_bc_upper = dirichlet_upper_BC
    else:
        raise ValueError("Given BC type is not avaliable!")

    state = pykinetic.State(domain, vmesh, 4)
    state.problem_data["v"] = vmesh.centers
    state.problem_data["phi"] = phi(solver.kn)
    state.problem_data["sqrt_phi"] = np.sqrt(phi(solver.kn))

    qinit(state, vmesh, solver.kn, init_func)
    sol = pykinetic.Solution(state, domain)

    output_dict = {}
    sol_frames, macro_frames, ts = (
        [copy.deepcopy(sol)],
        [compute_rho(sol.state, vmesh)],
        [0.0],
    )
    pbar = Progbar(nt)
    for t in range(nt):
        solver.evolve_to_time(sol)
        sol_frames.append(copy.deepcopy(sol))
        macro_frames.append(compute_rho(sol.state, vmesh))
        ts.append(0.0 + (t + 1) * dt)
        pbar.update(t + 1, finalize=False)
    pbar.update(nt, finalize=True)

    output_dict["macro_frames"] = macro_frames
    output_dict["x"] = x.centers
    output_dict["t"] = ts

    return output_dict
