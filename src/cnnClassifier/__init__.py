"""
Script name : __init__.py
Author name : Amine MEKKI
Date : 08/31/2023

Description : This script is used to initialize the logger for the project.
"""

import os 
import sys
import logging 
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), 'logs', LOG_FILE[:-4])

os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
logging_str = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    format = logging_str,
    level= logging.INFO, 
    handlers = [
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
    )

logger = logging.getLogger("cnnClassifierLogger")