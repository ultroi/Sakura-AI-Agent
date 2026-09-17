import logging


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger("sakura")
