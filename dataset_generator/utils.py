import sys
import logging
from pathlib import Path


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    # Suppress verbose logs from external libraries
    if not verbose:
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('datasets').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)


def create_output_directory(output_path: str) -> Path:
    """Create output directory if it doesn't exist."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
