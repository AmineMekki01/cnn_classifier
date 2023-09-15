from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion

STAGE_NAME = "Data Ingestion"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_file()
        
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
        data_ingestor = DataIngestionTrainingPipeline()
        data_ingestor.main()
        logger.info(f">>>>> Stage {STAGE_NAME} completed. <<<<< \n")
        
    except Exception as e:
        logger.exception(e)
        raise e

        