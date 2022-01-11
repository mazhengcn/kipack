import ml_collections


def validate_keys(base_cfg, config, base_filename="base_config.py"):
    """Validates that the config "inherits" from a base config.
    Args:
      base_cfg (`ConfigDict`): base config object containing the required fields
        for each experiment config.
      config (`ConfigDict`): experiment config to be checked against base_cfg.
      base_filename (str): file used to generate base_cfg.
    Raises:
      ValueError: if base_cfg contains keys that are not present in config.
    """

    for key in base_cfg.keys():
        if key not in config:
            raise ValueError(
                f"Key {key!r} missing from config. This config is required "
                f"to have keys: {list(base_cfg.keys())}. See {base_filename} for details."
            )
        if (
            isinstance(base_cfg[key], ml_collections.ConfigDict)
            and config[key] is not None
        ):
            validate_keys(base_cfg[key], config[key])


def validate_config(config):
    validate_keys(get_base_config(), config)


VELOCITY_MESH = {
    "nv": 64,
    "nr": 32,
    "s": 3.5,
    "quad_rule": "legendre",
    "nphi": 16,
    "ssrule": "womersley",
    "nsphr": 12,
    "dev": 7.0,
    "cmax": 0.0,
    "Tmax": 223.0,
    "lower": -1.0,
    "upper": 1.0,
    "radius": 1.0,
}

COLLISION_MODEL = {
    "dim": 2,
    "model_type": "vhs",
    "e": 1.0,
    "gamma": 0.0,
    "precision": "double",
    "xlo": -15e-3,
    "xhi": 15e-3,
    "Ne": 8,
    "H0": 30e-3,
    # non-dim
    "T0": 223,
    "rho0": 1.91607e-5,
    "molarMass0": 4.0047236985e-3,
}


def get_base_config():
    config = ml_collections.ConfigDict()

    config.velocity_mesh = ml_collections.ConfigDict(VELOCITY_MESH)
    config.collision_model = ml_collections.ConfigDict(COLLISION_MODEL)

    return config


if __name__ == "__main__":
    cfg = get_base_config()
    print(cfg.vmesh.nr)
