import numpy as np

EPS = 1e-8


def diff_x_up(psi):
    return psi[1:-1] - psi[:-2]


def diff_x_down(psi):
    return psi[2:] - psi[1:-1]

# 2nd order flux using combination of F_L (upwind) and F_H (Lax-Wendroff) and van_leer_limiter (can be changed to others).


def van_leer_limiter(r):
    return (r + np.abs(r))/(1. + np.abs(r))

# flux for upwind direction


def F_p_2(f, vp, dx, dt, limiter=van_leer_limiter):
    r = (f[1:-1] - f[:-2])/(f[2:] - f[1:-1] + EPS)
    phi = limiter(r)
    F = f.copy()
    F[1:-1] += 0.5*phi*(1. - vp*dt/dx)*(f[2:] - f[1:-1])
    return F[2:-2] - F[1:-3]

# flux for downwind direction


def F_m_2(f, vm, dx, dt, limiter=van_leer_limiter):
    r = (f[2:] - f[1:-1])/(f[1:-1] - f[:-2] + EPS)
    phi = limiter(r)
    F = f.copy()
    F[1:-1] += 0.5*phi*(-1. - vm*dt/dx)*(f[1:-1] - f[:-2])
    return F[3:-1] - F[2:-2]


def grad_vf2(f, v, dx, dt):
    vp = np.maximum(v, 0.)
    vm = np.minimum(v, 0.)
    # v*grad_f
    return vp[2:-2]*F_p_2(f, vp[1:-1], dx, dt) + vm[2:-2]*F_m_2(f, vm[1:-1], dx, dt)


def grad_vf1(f, v):
    vp = np.maximum(v, 0)
    vm = np.minimum(v, 0)
    return diff_x_up(vp*f) + diff_x_down(vm*f)
