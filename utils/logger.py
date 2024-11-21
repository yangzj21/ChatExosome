import logging
import datetime
import os

def init_basic_logger(log_file):
    log_file = log_file if log_file is not None else f"{datetime.now().strftime('%Y%m%d')}.log"
    lgr = logging.getLogger()

    different_log_file = False
    for handler in lgr.handlers:
        if isinstance(handler, logging.FileHandler):
            if os.path.abspath(handler.baseFilename) != os.path.abspath(log_file):
                different_log_file = True
    if different_log_file:
        lgr.handlers.clear()

    if not lgr.hasHandlers():
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s', datefmt='%H:%M:%S'))
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(message)s'))

        lgr.addHandler(file_handler)
        lgr.addHandler(console_handler)
        lgr.setLevel(logging.INFO)

    return lgr


def init_logger(ds_name):
    os.makedirs(f"./logs/", exist_ok=True)
    log_file = f"./logs/{ds_name}-{datetime.now().strftime('%Y%m%d')}.log"
    logger = init_basic_logger(log_file)
    return logger