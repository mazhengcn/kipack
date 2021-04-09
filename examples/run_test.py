if __name__ == "__main__":
    import cupy as cp
    import numpy as np

    from linear_transport.rbm_1d import run

    # Parameters

    def kn(x):
        return 0.05

    def sigma_s(x):
        return 1.0

    def sigma_a(x):
        return 0.0

    def Q(x):
        return 0.0

    xmin = 0.0
    xmax = 11.0
    nx = 1100
    dx = (xmax - xmin) / nx
    dt = 1e-4
    nt = 1000000
    BC = "dirichlet"

    def f_l(v):
        return 5.0 * np.sin(v)

    def f_r(v):
        return 0.0

    scheme = "Euler"

    def init_func(vmesh, rho, u, T):
        return 0.0

    with cp.cuda.Device(1):
        output_ref = run(
            kn=kn,
            sigma_s=sigma_s,
            sigma_a=sigma_a,
            Q=Q,
            xmin=xmin,
            xmax=xmax,
            nx=nx,
            dt=dt,
            nt=nt,
            BC=BC,
            f_l=f_l,
            f_r=f_r,
            coll="linear",
            scheme=scheme,
            init_func=init_func,
        )

    data_path = "./data/ref.npz"
    np.savez_compressed(data_path, **output_ref)
