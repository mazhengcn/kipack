import ml_collections

from kipack.collision.base_config import get_base_config

CONFIG_DIFFS = {
    "2d": {
        "collision_model": {"dim": 2, "e": 1, "gamma": 0},
        "velocity_mesh": {"nv": 64, "s": 3.5, "nphi": 16},
    },
    "3d": {
        "collision_model": {"dim": 3, "e": 1, "gamma": 0},
        "velocity_mesh": {"nv": 32, "s": 4.0, "nphi": 16},
    },
}


def get_config(name: str) -> ml_collections.ConfigDict:
    # Get base config
    cfg = get_base_config()
    # Update with selected name
    cfg.update(CONFIG_DIFFS[name])
    return cfg
