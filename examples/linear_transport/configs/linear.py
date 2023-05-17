import ml_collections

from kipack.collision.base_config import get_base_config

CONFIG_DIFFS = {
    "1d": {
        "collision_model": {"dim": 1, "e": 1, "gamma": 0},
        "velocity_mesh": {
            "nv": 60,
            "quad_rule": "uniform",
            "lower": -1.0,
            "upper": 1.0,
        },
    },
    "2d": {
        "collision_model": {"dim": 2, "e": 1, "gamma": 0},
        "velocity_mesh": {
            "nv": 32,
            "quad_rule": "uniform",
            "lower": 0.0,
            "upper": 6.283185307179586,
            "radius": 1.0,
        },
    },
}


def get_config(name: str) -> ml_collections.ConfigDict:
    # Get base config
    cfg = get_base_config()
    # Update with selected name
    cfg.update(CONFIG_DIFFS[name])
    return cfg
