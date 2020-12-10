import numpy as np

num_eqn = 1
num_waves = 1


def advection_1D(q_l, q_r, aux_l, aux_r, problem_data):
    # Number of Riemann problems we are solving
    num_rp = q_l.shape[1]
    num_vnodes = q_l.shape[2:]

    # Return values
    wave = np.empty((num_eqn, num_waves, num_rp) + num_vnodes)
    s = np.empty((num_waves, 1) + num_vnodes)
    amdq = np.zeros((num_eqn, num_rp) + num_vnodes)
    apdq = np.zeros((num_eqn, num_rp) + num_vnodes)

    wave[0, 0, :] = q_r[0, :] - q_l[0, :]
    s[0, 0, :] = problem_data["v"][0]

    apdq[0, :] = np.maximum(s[0, 0, :], 0.0) * wave[0, 0, :]
    amdq[0, :] = np.minimum(s[0, 0, :], 0.0) * wave[0, 0, :]

    return wave, s, amdq, apdq
