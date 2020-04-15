import json
import logging


LOGGING_LEVELS = {0: logging.CRITICAL,
                  1: logging.ERROR,
                  2: logging.WARNING,
                  3: logging.INFO,
                  4: logging.DEBUG}


class Params(object):
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance
        by `params.dict['learning_rate']`
        """
        return self.__dict__


# -----------------------------
class FrameCounter(object):
    # -----------------------------
    r"""
    Simple frame counter

    Simple frame counter to keep track of current frame number.  This can
    also be used to keep multiple runs frames seperated by having multiple
    counters at once.

    Initializes to 0
    """

    def __init__(self):
        self.__frame = 0

    def __repr__(self):
        return str(self.__frame)

    def increment(self):
        r"""
        Increment the counter by one
        """
        self.__frame += 1

    def set_counter(self, new_frame_num):
        r"""
        Set the counter to new_frame_num
        """
        self.__frame = new_frame_num

    def get_counter(self):
        r"""
        Get the current frame number
        """
        return self.__frame

    def reset_counter(self):
        r"""
        Reset the counter to 0
        """
        self.__frame = 0
