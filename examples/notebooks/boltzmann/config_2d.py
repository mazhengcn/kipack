from kipack.collision.base_config import get_base_config


def get_config():
    cfg = get_base_config()
    cfg.update(
        {
            "collision_model": {"dim": 2, "e": 1, "gamma": 0},
            "velocity_mesh": {"nv": 64, "s": 3.5, "nphi": 16},
        }
    )

    return cfg
