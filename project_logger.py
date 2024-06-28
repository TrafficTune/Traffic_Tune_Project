import logging
import sys
class ProjectLogger:
    def __init__(self, logger_name):
        self.logger_name = logger_name

    def setup_logger(self):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.INFO)

        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File Handler
        file_handler = logging.FileHandler(f"Logs/{self.logger_name}.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger