import copy
import math

import numpy as np

from examples.boltzmann.euler_1d import Euler1D
from kipack import collision, pykinetic
from kipack.utils import Progbar


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

    return rho / (2 * math.pi * T) ** (vdim / 2) * np.exp(-(v_u**2) / 2 / T)


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


class SplittingSolver1D(pykinetic.BoltzmannSolver1D):
    def __init__(self, riemann_solver=None, collision_operator=None, **kwargs):
        super().__init__(riemann_solver, collision_operator, **kwargs)

    def dq(self, state):
        deltaq = self.dq_hyperbolic(state)
        state.q += deltaq

        deltaq = self.dq_collision(state)

        return deltaq


def run(
    kn=1e-4,
    tau=None,
    dt=0.001,
    nt=100,
    eps=(0.2, 2.0),
    coll="fsm",
    scheme="Euler",
    euler_solver=False,
):
    config = collision.utils.CollisionConfig.from_json("./configs/" + coll + ".json")

    vmesh = collision.SpectralMesh(config)
    if coll == "fsm":
        coll_op = collision.FSInelasticVHSCollision(config, vmesh)
    elif coll == "rbm":
        coll_op = collision.RandomBatchCollisionV2(config, vmesh)
        a, b = eps
        coll_op.eps = a * vmesh.delta**b
    else:
        raise NotImplementedError(
            "Collision method {} is not implemented.".format(coll)
        )

    rp = pykinetic.riemann.advection_1D
    solver = pykinetic.BoltzmannSolver1D(
        rp, coll_op, kn=kn, heat_bath=tau, device="gpu"
    )
    if "RK" in scheme:
        solver.time_integrator = "RK"
        solver.a = rkcoeff[scheme]["a"]
        solver.b = rkcoeff[scheme]["b"]
        solver.c = rkcoeff[scheme]["c"]
    else:
        solver.time_integrator = scheme
    solver.dt = dt
    print("dt is {}".format(solver.dt))
    solver.order = 2
    solver.lim_type = -1
    solver.bc_lower[0] = pykinetic.BC.periodic
    solver.bc_upper[0] = pykinetic.BC.periodic

    x = pykinetic.Dimension(0.0, 1.0, 100, name="x")
    domain = pykinetic.Domain([x])
    state = pykinetic.State(domain, vdof=vmesh.nvs)
    state.problem_data["v"] = vmesh.centers
    qinit(state, vmesh)
    sol = pykinetic.Solution(state, domain)

    output_dict = {}

    if euler_solver:
        euler = Euler1D(tau)
        euler.set_initial(vmesh.get_p(sol.q)[0], 0, 1.0)
        euler.solve(dt * nt)
        output_dict["euler_macro"] = euler.macros(2)

    sol_frames = []
    macro_frames = []
    pbar = Progbar(nt)
    for t in range(nt):
        solver.evolve_to_time(sol)
        # l2_err = (
        #     np.sqrt(np.sum((sol.q - bkw_fn(vmesh, sol.t)) ** 2))
        # * vmesh.delta
        # )
        # sol_frames.append([copy.deepcopy(sol), l2_err])
        macro_frames.append(vmesh.get_p(sol.q))
        sol_frames.append(copy.deepcopy(sol))
        pbar.update(t + 1, finalize=False)
    pbar.update(nt, finalize=True)

    output_dict["macro_frames"] = macro_frames

    return output_dict


def qinit(state, vmesh):
    x = state.grid.x.centers
    rho = 1.0 + 0.5 * np.cos(4 * math.pi * x)
    state.q[:] = maxwellian_vec_init(vmesh, 0.0, 1.0, rho)


def maxwellian_vec_init(vmesh, u, T, rho):
    v = vmesh.center
    return (
        rho[:, None, None]
        / (2 * math.pi * T)
        * np.exp(-((v - u)[:, None] ** 2 + v**2) / (2 * T))
    )


def bkw_fn(vmesh, t):
    vsq = vmesh.vsquare

    K = 1 - 0.5 * np.exp(-t / 8)
    return (
        1
        / (2 * math.pi * K**2)
        * np.exp(-0.5 * vsq / K)
        * (2 * K - 1 + 0.5 * vsq * (1 - K) / K)
    )


def ext_Q(vmesh, t):
    vsq = vmesh.vsquare

    K = 1 - np.exp(-t / 8) / 2
    dK = np.exp(-t / 8) / 16
    df = (-2 / K + vsq / (2 * K**2)) * bkw_fn(vmesh, t) + 1 / (
        2 * math.pi * K**2
    ) * np.exp(-vsq / (2 * K)) * (2 - vsq / (2 * K**2))
    return df * dK


def ext_T(rho0, T0, e, tau, t):
    # exact evolution of temperature
    # assume u0 = 0
    return (T0 - 8 * tau / (1 - e**2)) * np.exp(
        -(1 - e**2) * t * rho0 / 4
    ) + 8 * tau / (1 - e**2)


def flat(vmesh, T0):
    vx, vy = vmesh.centers
    w = np.sqrt(3 * T0)
    return 1 / 4 / w**2 * (vx <= w) * (vx >= -w) * (vy <= w) * (vy >= -w)


def maxwellian_init(vmesh, K):
    return 1 / (2 * math.pi * K) * np.exp(-0.5 * vmesh.vsq / K)


def anisotropic_f(v):
    return (
        0.8
        * math.pi ** (-1.5)
        * (
            np.exp(
                -(16 ** (1 / 3))
                * ((v - 2)[:, None, None] ** 2 + (v - 2)[:, None] ** 2 + (v - 2) ** 2)
            )
            + np.exp(
                -(v + 0.5)[:, None, None] ** 2
                - (v + 0.5)[:, None] ** 2
                - (v + 0.5) ** 2
            )
        )
    )
