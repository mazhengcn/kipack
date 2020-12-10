import copy
import math

import numpy as np
from examples.utils import Progbar
from kipack import collision, pykinetic
from kipack.pykinetic.boltzmann.solver import BoltzmannSolver1D


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
    if init_func:
        pos_v = init_func(v, 0.0, 1.0, rho)
        # print(pos_v)
        min_v = init_func(-v, 0.0, 1.0, rho)
        state.q[0, :] = 0.5 * (pos_v + min_v)
        state.q[1, :] = 0.5 / kn * (pos_v - min_v)
    else:
        state.q[:] = 0.0


def sigma(x):
    # return (x < 0.5) * 2.0 + (x >= 0.5) * 1.0
    return np.ones(x.shape) * 1.0


def compute_rho(state, vmesh):
    vaxis = tuple(-(i + 1) for i in range(vmesh.num_dim))
    return np.sum(state.q[0, ...] * vmesh.weights, axis=vaxis)


class APNeutronTransportSolver1D(BoltzmannSolver1D):
    def step_collision(self, state):
        dt_kn2 = self.dt / self.kn ** 2

        # Update r
        rho = (
            self.coll[0](state.q[0, :], heat_bath=self.tau, device=self.device)
            + state.q[0, :]
        ) * 2
        state.q[0, :] = (dt_kn2 * rho + state.q[0, :]) / (1.0 + dt_kn2)

        # Compute r_x
        num_ghost = self.num_ghost
        self._apply_bcs(state)
        rbc = self.qbc[0, :]
        dr_dx = (
            rbc[num_ghost + 1 : -num_ghost + 1, :]
            - rbc[num_ghost - 1 : -num_ghost - 1, :]
        )
        dr_dx /= 2 * state.grid.delta[0]

        # Update j
        phi = state.problem_data["phi"]
        v = state.problem_data["v"][0]
        state.q[1, :] = (
            state.q[1, :] - dt_kn2 * (1.0 - self.kn ** 2 * phi) * v * dr_dx
        )
        state.q[1, :] /= 1.0 + dt_kn2

    def dq(self, state):
        self.step_collision(state)
        deltaq = self.dq_hyperbolic(state)
        return deltaq


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
        "./examples/linear_transport/configs/" + "parity" + ".json"
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
    x = pykinetic.Dimension(0.0, 1.0, 40, name="x")
    domain = pykinetic.Domain([x])

    # Riemann solver
    rp = pykinetic.riemann.parity_1D
    solver = APNeutronTransportSolver1D(rp, [coll_op], kn=kn)
    # print(solver.kn)
    solver.order = 2
    # solver.lim_type = -1
    solver.time_integrator = scheme
    solver.dt = dt
    print("dt is {}".format(solver.dt))

    fL = 1.0
    fR = 0.0

    # Boundary conditions
    def dirichlet_lower_BC(state, dim, t, qbc, auxbc, num_ghost):
        v = state.problem_data["v"][0]
        kn_dx = kn / state.grid.delta[0]
        for i in range(num_ghost):
            qbc[0, i, :] = (fL - (0.5 - kn_dx * v) * qbc[0, num_ghost, :]) / (
                0.5 + kn_dx * v
            )
        for i in range(num_ghost):
            qbc[1, i, :] = (
                2 * fL - (qbc[0, num_ghost - 1, :] + qbc[0, num_ghost, :])
            ) / kn - qbc[1, num_ghost, :]

    def dirichlet_upper_BC(state, dim, t, qbc, auxbc, num_ghost):
        v = state.problem_data["v"][0]
        kn_dx = kn / state.grid.delta[0]
        for i in range(num_ghost):
            qbc[0, -i - 1, :] = (
                fR - (0.5 - kn_dx * v) * qbc[0, -num_ghost - 1, :]
            ) / (0.5 + kn_dx * v)
        for i in range(num_ghost):
            qbc[1, -i - 1, :] = (
                (qbc[0, -num_ghost, :] + qbc[0, -num_ghost - 1, :]) - 2 * fR
            ) / kn - qbc[1, -num_ghost - 1, :]

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

    state = pykinetic.State(domain, vmesh, 2)
    state.problem_data["v"] = vmesh.centers
    state.problem_data["phi"] = phi(kn)
    state.problem_data["sqrt_phi"] = np.sqrt(phi(kn))
    print(state.grid.delta[0])

    qinit(state, vmesh, kn, init_func)
    sol = pykinetic.Solution(state, domain)

    output_dict = {}
    sol_frames = []
    macro_frames = []
    pbar = Progbar(nt)
    for t in range(nt):
        macro_frames.append(compute_rho(sol.state, vmesh))
        solver.evolve_to_time(sol)
        sol_frames.append(copy.deepcopy(sol))
        pbar.update(t + 1, finalize=False)
    pbar.update(nt, finalize=True)
    macro_frames.append(compute_rho(sol.state, vmesh))

    output_dict["macro_frames"] = macro_frames

    return output_dict
