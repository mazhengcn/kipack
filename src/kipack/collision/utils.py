import dataclasses

from kipack.params import Config


@dataclasses.dataclass
class VMeshConfig(Config):
    nv: int = 64
    s: float = 3.5
    quad_rule: str = "legendre"
    nr: int = nv // 2
    nphi: int = 16
    ssrule: str = "womersley"
    nsphr: int = 12
    dev: float = 7.0
    cmax: float = 0.0
    Tmax: float = 223.0


@dataclasses.dataclass
class CollisionModelConfig(Config):
    dim: int = 2
    model_type: str = "vhs"
    e: float = 1.0
    gamma: float = 0.0


@dataclasses.dataclass
class CollisionConfig(Config):
    precision: str = "double"
    # mesh
    xlo: float = -15e-3
    xhi: float = 15e-3
    Ne: float = 8
    H0: float = 30e-3
    # non-dim
    T0: float = 223
    rho0: float = 1.91607e-5
    molarMass0: float = 4.0047236985e-3

    velocity_mesh: VMeshConfig = VMeshConfig()
    collision_model: CollisionModelConfig = CollisionModelConfig()
