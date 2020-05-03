import logging


LOGGING_LEVELS = {
    0: logging.CRITICAL,
    1: logging.ERROR,
    2: logging.WARNING,
    3: logging.INFO,
    4: logging.DEBUG,
}


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
