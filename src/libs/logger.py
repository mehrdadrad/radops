import logging
from config.config import settings


def initialize_logger() -> None:
    """Configures the root logger and specific library loggers."""
    level = settings.logging.level.upper()
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("redisvl").setLevel(logging.WARNING)
