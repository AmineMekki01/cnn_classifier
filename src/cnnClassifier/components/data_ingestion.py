import os
import urllib.request as request
import zipfile
import smtplib
from base64 import b64encode
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.utils.common_functions import get_size
from cnnClassifier import logger
from pathlib import Path
class DataIngestion:
    MAX_RETRIES = 3

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            file_name, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{file_name.split('/')[-1]} downloaded! with following info : \n {headers}")
        else:
            logger.info(f"File already exists of size : {get_size(Path(self.config.local_data_file))}")
            
    def extract_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)

    
    
    # potential function to be added 
    
    def validate_data(self):
        if os.path.getsize(self.config.local_data_file) == 0:
            raise ValueError("Downloaded file is empty")

    def notify_failure(self, message):
        # Email Notification
        with smtplib.SMTP('smtp.example.com', 587) as server:
            server.login("Email", "password")
            server.sendmail("sender email", "receiver email", message)

    def format_DI_error_mail(self, subject, message): # data ingestion email error format 
        body = "There was an error during data ingestion. Here's the error message:\n\n" + message
        formatted_message = f"Subject: {subject}\n\n{body}"
        return formatted_message

