import logging

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.propagate = False   # Prevent duplicate logs

if not logger.handlers:

    consoleHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler('error.txt')

    consoleHandler.setLevel(logging.DEBUG)
    fileHandler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)