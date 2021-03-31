import copy
import math

import numpy as np
from examples.utils import Progbar
from kipack import collision, pykinetic
from kipack.pykinetic.boltzmann.solver import BoltzmannSolver1D

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


def maxwellian_vec_init(vmesh, u, T, rho):
    v = vmesh.center
    return (
        rho[:, None]
        / np.sqrt(2 * math.pi * T)
        * np.exp(-((v - u) ** 2) / (2 * T))
    )


def qinit(state, vmesh, init_func):
    x = state.grid.x.centers
    rho = 1.0 + 2.0 * np.cos(4 * math.pi * x)
    state.q[0, ...] = init_func(vmesh, 0.0, 1.0, rho)


def compute_rho(state, vmesh):
    vaxis = tuple(-(i + 1) for i in range(vmesh.num_dim))
    return np.sum(state.q[0, ...] * vmesh.weights, axis=vaxis)


class DiffusiveRegimeSolver1D(BoltzmannSolver1D):
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

    def dq(self, state):
        deltaq = self.dq_hyperbolic(state) / self.kn
        deltaq += self.dq_collision(state) / self.kn * self.sigma_s
        deltaq -= self.sigma_a * state.q * self.dt
        deltaq += self.Q * self.dt
        return deltaq


def run(
    kn=1.0,
    sigma_s=lambda x: 1.0,
    sigma_a=lambda x: 0.0,
    Q=lambda x: 0.0,
    xmin=0.0,
    xmax=1.0,
    nx=40,
    dt=0.01,
    nt=1000,
    coll="linear",
    scheme="Euler",
    BC="periodic",
    f_l=lambda v: 1.0,
    f_r=lambda v: 0.0,
    init_func=lambda vmesh, u, T, rho: 0.0,
):
    # Load config
    config = collision.utils.CollisionConfig.from_json(
        "./examples/linear_transport/configs/" + "linear" + ".json"
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
    x = pykinetic.Dimension(xmin, xmax, nx, name="x")
    domain = pykinetic.Domain([x])

    # Riemann solver
    rp = pykinetic.riemann.advection_1D
    solver = DiffusiveRegimeSolver1D(
        rp,
        [coll_op],
        kn=kn(x.centers),
        sigma_s=sigma_s(x.centers),
        sigma_a=sigma_a(x.centers),
        Q=Q(x.centers),
    )
    solver.order = 1
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
            qbc[0, i, v > 0] = f_l(v[v > 0])

    def dirichlet_upper_BC(state, dim, t, qbc, auxbc, num_ghost):
        v = state.problem_data["v"][0]
        for i in range(num_ghost):
            qbc[0, -i - 1, v < 0] = f_r(v[v < 0])

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
