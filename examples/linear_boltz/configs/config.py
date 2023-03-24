import ml_collections

from kipack.collision.base_config import get_base_config

CONFIG_DIFFS = {
    "linear": {
        "collision_model": {
            "dim": 1,
            "e": 1.0,
            "gamma": 0.0
        },
        "velocity_mesh": {
            "nv": 30,
            "quad_rule": "hermite",
            "lower": 0.0,
            "upper": 1.7724538509055159
        }
    },
}


def get_config(name: str) -> ml_collections.ConfigDict:
    # Get base config
    cfg = get_base_config()
    # Update with selected name
    cfg.update(CONFIG_DIFFS[name])
    return cfg
