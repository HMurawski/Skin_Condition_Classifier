import os, sys, logging
from logging.handlers import RotatingFileHandler

def init_logger(name: str = "skinclf", level: int = logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  

    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    os.makedirs("logs", exist_ok=True)
    fh = RotatingFileHandler("logs/app.log", maxBytes=1_000_000, backupCount=2)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
