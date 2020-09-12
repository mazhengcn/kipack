import numpy as np


def advection_1D(q_l, q_r, aux_l, aux_r, problem_data):
    r"""Basic 1d advection riemann solver
    *problem_data* should contain -
     - *vx* - (float) Determines advection speed
    """

    # wave shape: q.shape = (nx, nv, nv) (2D)
    wave = q_r - q_l
    # s shape: v shape = (2, nv, nv) (2D)
    s = problem_data["v"][0]
    apdq = np.maximum(s, 0.0) * wave
    amdq = np.minimum(s, 0.0) * wave

    return wave, s, amdq, apdq


def advection_1D_well_balanced(q_l, q_r, aux_l, aux_r, problem_data):
    r"""Basic 1d advection riemann solver
    *problem_data* should contain -
     - *vx* - (float) Determines advection speed
    """

    # wave shape: q.shape = (nx, nv, nv) (2D)
    wave = q_r - q_l
    # s shape: v shape = (2, nv, nv) (2D)
    s = problem_data["v"][0]
    apdq = np.maximum(s, 0.0) * wave
    amdq = np.minimum(s, 0.0) * wave

    apdq -= (s > 0.0) * 0.5 * (aux_r + aux_l)
    amdq -= (s < 0.0) * 0.5 * (aux_r + aux_l)

    return wave, s, amdq, apdq
