import logging


def setup_logger(name, level=logging.INFO) -> logging.Logger:
    """
    Function to set up a simple logger.

    :param name: Name of the logger.
    :param level: Logging level (default is INFO).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create a formatter and set it for handler
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)

    return logger
