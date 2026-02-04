import logging
from logging.handlers import RotatingFileHandler
from config.config import settings


def _parse_size(size_str: str) -> int:
    """Parses a size string (e.g., '10 MB') into bytes."""
    units = {"GB": 1024**3, "MB": 1024**2, "KB": 1024, "B": 1}
    size_str = size_str.upper().strip()
    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                number = float(size_str[:-len(unit)].strip())
                return int(number * multiplier)
            except ValueError:
                break
    return 10 * 1024 * 1024  # Default to 10 MB


def initialize_logger() -> None:
    """Configures the root logger and specific library loggers."""
    level = settings.logging.level.upper()
    log_file = settings.logging.file

    handlers = []
    if log_file:
        max_bytes = _parse_size(settings.logging.rotation)
        # backupCount=5 keeps 5 old log files (retention by count)
        handlers.append(
            RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=5)
        )
    else:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("redisvl").setLevel(logging.WARNING)
