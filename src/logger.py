import logging
import os


def get_logger(name):
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        filename="logs/aqi_project.log",
        format='[%(levelname)s] - %(name)s - %(asctime)s - %(message)s',
        level=logging.INFO
    )

    return logging.getLogger(name)
    