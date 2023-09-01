from cnnClassifier import logger
from cnnClassifier.pipeline.S1_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.S2_prepare_base_model import PrepareBaseModelTrainingPipeline




# testing 
STAGE_NAME1 = "Data Ingestion"
STAGE_NAME2 = "Prepare Base Model"

if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage {STAGE_NAME1} started <<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>> Stage {STAGE_NAME1} completed. <<<<< \n")
        
    except Exception as e:
        logger.exception(e)
        
    try:
        logger.info(f">>>>> Stage {STAGE_NAME2} started <<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> Stage {STAGE_NAME2} completed. <<<<< \n")
    except Exception as e:
        logger.exception(e)