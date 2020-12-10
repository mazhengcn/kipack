import numpy as np

num_eqn = 2
num_waves = 2


def parity_1D(q_l, q_r, aux_l, aux_r, problem_data):
    # Convenience
    num_rp = q_l.shape[1]
    vshape = q_l.shape[2:]

    # Return values
    wave = np.empty((num_eqn, num_waves, num_rp) + vshape)
    s = np.empty((num_waves, 1) + vshape)
    amdq = np.empty((num_eqn, num_rp) + vshape)
    apdq = np.empty((num_eqn, num_rp) + vshape)

    # Local values
    delta = np.empty(np.shape(q_l))
    delta = q_r - q_l

    # Compute the waves
    # 1-Wave
    a1 = 0.5 * (delta[0, :] - delta[1, :] / problem_data["sqrt_phi"])
    wave[0, 0, :] = a1
    wave[1, 0, :] = -a1 * problem_data["sqrt_phi"]
    s[0, 0] = -problem_data["sqrt_phi"] * problem_data["v"][0]

    # 2-Wave
    a2 = 0.5 * (delta[0, :] + delta[1, :] / problem_data["sqrt_phi"])
    wave[0, 1, :] = a2
    wave[1, 1, :] = a2 * problem_data["sqrt_phi"]
    s[1, 0] = problem_data["sqrt_phi"] * problem_data["v"][0]

    # Compute the left going and right going fluctuations
    for m in range(num_eqn):
        amdq[m, :] = s[0, :] * wave[m, 0, :]
        apdq[m, :] = s[1, :] * wave[m, 1, :]

    return wave, s, amdq, apdq
