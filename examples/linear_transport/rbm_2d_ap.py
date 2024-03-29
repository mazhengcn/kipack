import copy
import math

import numpy as np
from examples.utils import Progbar
from kipack import collision, pykinetic
from kipack.pykinetic.boltzmann.solver import BoltzmannSolver2D


def phi(eps):
    return np.minimum(1.0, 1.0 / eps)


def maxwellian_vec_init(vx, vy, ux, uy, T, rho):
    return (
        rho[..., None, None]
        / (2 * math.pi * T)
        * np.exp(-((vx - ux) ** 2 - (vy - uy) ** 2) / (2 * T))
    )


def qinit(state, vmesh, kn, init_func=None):
    x, y = state.grid.x.centers, state.grid.y.centers
    rho = 1.0 + 2.0 * np.exp((-x[..., None] ** 2 - y ** 2) / 0.25)
    vx, vy = vmesh.centers

    pvx_mvy = init_func(vx, -vy, 0.0, 0.0, 0.25, rho)
    mvx_pvy = init_func(-vx, vy, 0.0, 0.0, 0.25, rho)
    pvx_pvy = init_func(vx, vy, 0.0, 0.0, 0.25, rho)
    mvx_mvy = init_func(-vx, -vy, 0.0, 0.0, 0.25, rho)

    state.q[0, :] = 0.5 * (pvx_mvy + mvx_pvy)
    state.q[1, :] = 0.5 / kn * (pvx_mvy - mvx_pvy)
    state.q[2, :] = 0.5 * (pvx_pvy + mvx_mvy)
    state.q[3, :] = 0.5 / kn * (pvx_pvy - mvx_mvy)


def compute_rho(state, vmesh):
    vaxis = tuple(-(i + 1) for i in range(vmesh.num_dim))
    rho = np.sum(state.q[::2] * vmesh.weights, axis=vaxis)
    return 0.5 * (rho[0] + rho[1])


class APNeutronTransportSolver2D(BoltzmannSolver2D):
    def __init__(
        self, riemann_solver, collision_operator, kn, sigma_s, sigma_a, Q, **kwargs
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
        rho = 0.5 * (rho[0] + rho[1])
        state.q[::2] = (dt_kn2 * self.sigma_s * rho + state.q[::2]) / (
            1.0 + dt_kn2 * self.sigma_s
        )

        vx, vy = state.problem_data["v"]
        # Compute vx * dr / dx
        num_ghost = self.num_ghost
        self._apply_bcs(state)
        rbc = self.qbc[::2]
        vxdr_dx = vx * (
            rbc[:, num_ghost + 1 : -num_ghost + 1, num_ghost:-num_ghost]
            - rbc[:, num_ghost - 1 : -num_ghost - 1, num_ghost:-num_ghost]
        )
        vxdr_dx /= 2 * state.grid.delta[0]
        # Compute vy * dr / dy
        vydr_dy = vy * (
            rbc[:, num_ghost:-num_ghost, num_ghost + 1 : -num_ghost + 1]
            - rbc[:, num_ghost:-num_ghost, num_ghost - 1 : -num_ghost - 1]
        )
        vydr_dy /= 2 * state.grid.delta[1]
        vydr_dy[0, :] = -vydr_dy[0, :]

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
        "./examples/linear_transport/configs/" + "parity_2d" + ".json"
    )

    # Collision
    vmesh = collision.PolarMesh(config)
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
    rp = pykinetic.riemann.parity_2D
    solver = APNeutronTransportSolver2D(
        rp,
        [coll_op],
        kn=kn(x.centers, y.centers),
        sigma_s=sigma_s(x.centers, y.centers),
        sigma_a=sigma_a(x.centers, y.centers),
        Q=Q(x.centers, y.centers),
    )
    # print(solver.kn)
    solver.order = 1
    # solver.lim_type = -1
    solver.time_integrator = scheme
    solver.dt = dt
    print("dt is {}".format(solver.dt))

    # sigma = sigma_s(x.centers, y.centers) + kn(
    #     x.centers, y.centers
    # ) ** 2 * sigma_a(x.centers, y.centers)
    # sigma_l, sigma_r = None, None
    # if isinstance(sigma, np.ndarray):
    #     sigma_l, sigma_r = sigma[0], sigma[-1]
    # else:
    #     sigma_l = sigma_r = sigma
    # kn_l, kn_r = None, None
    # if isinstance(solver.kn, np.ndarray):
    #     kn_l, kn_r = solver.kn[0], solver.kn[-1]
    # else:
    #     kn_l = kn_r = solver.kn

    # Boundary conditions
    def dirichlet_lower_BC(state, dim, t, qbc, auxbc, num_ghost):
        vx, vy = state.problem_data["v"]
        if dim.name == "x":
            for i in range(num_ghost):
                qbc[::2, i] = -qbc[::2, num_ghost + i]
                qbc[1::2, i] = (
                    2 * (-vx * 2 * qbc[::2, num_ghost] / state.grid.delta[0])
                    - qbc[1::2, num_ghost + i]
                )
        elif dim.name == "y":
            for i in range(num_ghost):
                qbc[::2, :, i] = -qbc[::2, :, num_ghost + i]
                qbc[1::2, :, i] = (
                    2 * (-vy * 2 * qbc[::2, :, num_ghost] / state.grid.delta[1])
                    - qbc[1::2, :, num_ghost + i]
                )
                qbc[1, :, i] = -qbc[1, :, i]
        else:
            raise ValueError("Dim could be only x or y.")

    def dirichlet_upper_BC(state, dim, t, qbc, auxbc, num_ghost):
        vx, vy = state.problem_data["v"]
        if dim.name == "x":
            for i in range(num_ghost):
                qbc[::2, -i - 1] = -qbc[::2, -2 * num_ghost + i]
                qbc[1::2, -i - 1] = (
                    2 * (vx * 2 * qbc[::2, -num_ghost - 1] / state.grid.delta[0])
                    - qbc[1::2, -2 * num_ghost + i]
                )
        elif dim.name == "y":
            for i in range(num_ghost):
                qbc[::2, :, -i - 1] = -qbc[::2, :, -2 * num_ghost + i]
                qbc[1::2, :, -i - 1] = (
                    2 * (vy * 2 * qbc[::2, :, -num_ghost - 1] / state.grid.delta[1])
                    - qbc[1::2, :, -2 * num_ghost + i]
                )
                qbc[1, :, -i - 1] = -qbc[1, :, -i - 1]
        else:
            raise ValueError("Dim could be only x or y.")

    if BC == "periodic":
        solver.all_bcs = pykinetic.BC.periodic
    elif BC == "dirichlet":
        solver.all_bcs = pykinetic.BC.custom
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
    output_dict["mesh"] = state.c_centers
    output_dict["t"] = ts

    return output_dict
