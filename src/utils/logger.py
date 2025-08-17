import logging
import sys

def get_logger(name: str) -> logging.Logger:
    r"""
    Gets a standard logger with a stream hander to stdout.
    """
    formatter = logging.Formatter(
        fmt="[%(levelname)s|%(module)s:%(lineno)d] %(asctime)s - [%(name)s] >> %(message)s", datefmt="%m-%d-%Y %H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger