from kipack.collision.base_config import get_base_config


def get_config():
    cfg = get_base_config()
    cfg.update(
        {
            "collision_model": {"dim": 3, "e": 1, "gamma": 0},
            "velocity_mesh": {"nv": 32, "s": 4.0, "nphi": 16},
        }
    )

    return cfg
