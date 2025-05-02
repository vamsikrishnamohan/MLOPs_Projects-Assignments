import logging

logging.basicConfig(filename="pipeline.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def log_info(message):
    """Logs an info message."""
    logging.info(message)

def log_error(error):
    """Logs an error message."""
    logging.error(error)
