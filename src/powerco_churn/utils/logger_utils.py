"""
This module contains functions to configure logging
"""

import logging
from pathlib import Path


def configure_logging(
    log_file_name: str = "app.log", 
    level: int = logging.INFO,
    project_root: Path = None,

):
    """
    Set up logging to both the console and a file. Logs are saved to the 'logs'
    directory at the root of the project, regardless of where the script
    is run.

    Args:
        log_file_name (str): Name of the log file (e.g., "wrangling.log").
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    logger = logging.getLogger()

    if logger.hasHandlers():
        return  # Avoid re-configuring if already set

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
     # Determine project root
    if project_root is None:
        project_root = Path(__file__).resolve()
        while (project_root.name != 'powerco_churn' and 
                            project_root != project_root.parent):
            project_root = project_root.parent

    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_dir / log_file_name

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
