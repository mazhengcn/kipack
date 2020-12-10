import copy
import math

import numpy as np
from kipack import collision, pykinetic
from kipack.pykinetic.boltzmann.solver import BoltzmannSolver1D
from utils import Progbar


class DiffusiveRegimeSolver1D(BoltzmannSolver1D):
    def dq(self, state):
        deltaq = self.dq_hyperbolic(state) / self.kn
        deltaq += self.dq_collision(state) / self.kn
        return deltaq


rkcoeff = {
    "RK3": {
        "a": np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0], [-1.0, 2.0, 0.0]]),
        "b": np.array([1 / 6, 2 / 3, 1 / 6]),
        "c": np.array([0, 0.5, 1.0]),
    },
    "RK4": {
        "a": np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
        "b": np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]),
        "c": np.array([0.0, 0.5, 0.5, 1.0]),
    },
}


def maxwellian(v, rho, u, T):
    vdim, v_u = None, None
    if isinstance(rho, np.ndarray):
        v = np.asarray(v)
        u = np.asarray(u)
        xdim = rho.ndim
        vdim = v.shape[0]
        v_dim = np.index_exp[:] + xdim * (np.newaxis,)
        v = v[v_dim]
        q_dim = (...,) + vdim * (np.newaxis,)
        rho, u, T = rho[q_dim], u[q_dim], T[q_dim]
        v_u = np.sum((v - u) ** 2, axis=0)

    return rho / (2 * math.pi * T) ** (vdim / 2) * np.exp(-(v_u ** 2) / 2 / T)


def maxwellian_vec_init(vmesh, u, T, rho):
    v = vmesh.center
    return (
        rho[:, None]
        / np.sqrt(2 * math.pi * T)
        * np.exp(-((v - u) ** 2) / (2 * T))
    )


def bkw_fn(vmesh, t):
    vsq = vmesh.vsquare

    K = 1 - 0.5 * np.exp(-t / 8)
    return (
        1
        / (2 * math.pi * K ** 2)
        * np.exp(-0.5 * vsq / K)
        * (2 * K - 1 + 0.5 * vsq * (1 - K) / K)
    )


def flat(vmesh, T0):
    vx, vy = vmesh.centers
    w = np.sqrt(3 * T0)
    return 1 / 4 / w ** 2 * (vx <= w) * (vx >= -w) * (vy <= w) * (vy >= -w)


def maxwellian_init(vmesh, K):
    return 1 / (2 * math.pi * K) * np.exp(-0.5 * vmesh.vsq / K)


def anisotropic_f(v):
    return (
        0.8
        * math.pi ** (-1.5)
        * (
            np.exp(
                -(16 ** (1 / 3))
                * (
                    (v - 2)[:, None, None] ** 2
                    + (v - 2)[:, None] ** 2
                    + (v - 2) ** 2
                )
            )
            + np.exp(
                -(v + 0.5)[:, None, None] ** 2
                - (v + 0.5)[:, None] ** 2
                - (v + 0.5) ** 2
            )
        )
    )


def qinit(state, vmesh, init_func=None):
    x = state.grid.x.centers
    rho = 1.0 + 0.5 * np.cos(4 * math.pi * x)
    if init_func:
        state.q[0, ...] = init_func(vmesh, 0.0, 1.0, rho)
    else:
        state.q[0, ...] = 0.0


def sigma(x):
    # return (x < 0.5) * 2.0 + (x >= 0.5) * 1.0
    return np.ones(x.shape) * 1.0


def compute_rho(state, vmesh):
    vaxis = tuple(-(i + 1) for i in range(vmesh.num_dim))
    return 0.5 * np.sum(state.q[0, ...] * vmesh.weights, axis=vaxis)


def run(
    kn=1.0,
    dt=0.01,
    nt=1000,
    coll="linear",
    scheme="Euler",
    BC="periodic",
    init_func=None,
):
    # Load config
    config = collision.utils.CollisionConfig.from_json(
        "./configs/" + "linear" + ".json"
    )

    # Collision
    vmesh = collision.CartesianMesh(config)
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
    x = pykinetic.Dimension(0.0, 1.0, 100, name="x")
    domain = pykinetic.Domain([x])

    # Riemann solver
    rp = pykinetic.riemann.advection_1D
    solver = DiffusiveRegimeSolver1D(
        rp, coll_op, kn=kn / sigma(x.centers)[:, None]
    )
    solver.order = 2
    # solver.lim_type = 2
    # Time integrator
    if "RK" in scheme:
        solver.time_integrator = "RK"
        solver.a = rkcoeff[scheme]["a"]
        solver.b = rkcoeff[scheme]["b"]
        solver.c = rkcoeff[scheme]["c"]
    else:
        solver.time_integrator = scheme
    solver.dt = dt
    print("dt is {}".format(solver.dt))

    # Boundary condition
    def dirichlet_lower_BC(state, dim, t, qbc, auxbc, num_ghost):
        v = state.problem_data["v"][0]
        for i in range(num_ghost):
            qbc[0, i, v > 0] = 1.0 + np.cos(v)[v > 0]

    def dirichlet_upper_BC(state, dim, t, qbc, auxbc, num_ghost):
        v = state.problem_data["v"][0]
        for i in range(num_ghost):
            qbc[0, -i - 1, v < 0] = 0.0 + 2 * np.cos(v)[v < 0]

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

    state = pykinetic.State(domain, vmesh, 1)
    state.problem_data["v"] = vmesh.centers
    qinit(state, vmesh, init_func)
    sol = pykinetic.Solution(state, domain)

    output_dict = {}
    sol_frames = []
    macro_frames = []
    pbar = Progbar(nt)
    for t in range(nt):
        solver.evolve_to_time(sol)
        macro_frames.append(compute_rho(sol.state, vmesh))
        sol_frames.append(copy.deepcopy(sol))
        pbar.update(t + 1, finalize=False)
    pbar.update(nt, finalize=True)

    output_dict["macro_frames"] = macro_frames

    return output_dict
