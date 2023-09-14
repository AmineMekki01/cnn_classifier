from cnnClassifier import logger
from cnnClassifier.pipeline.S1_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.S2_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.S3_training import ModelTrainingPipeline
from cnnClassifier.pipeline.S4_evaluation import ModelEvaluationPipeline
from cnnClassifier.pipeline.S5_inferencer import ModelInferencePipeline




# testing 
STAGE_NAME1 = "Data Ingestion"
STAGE_NAME2 = "Prepare Base Model"
STAGE_NAME3 = "Training"
STAGE_NAME4 = "Evaluation"
STAGE_NAME5 = "Inference"


if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage {STAGE_NAME1} started <<<<<")
        data_ingestor = DataIngestionTrainingPipeline()
        data_ingestor.main()
        logger.info(f">>>>> Stage {STAGE_NAME1} completed. <<<<< \n")
        
    except Exception as e:
        logger.exception(e)
        raise e
    
    try:
        logger.info(f">>>>> Stage {STAGE_NAME2} started <<<<<")
        prepare_base_model = PrepareBaseModelTrainingPipeline()
        prepare_base_model.main()
        logger.info(f">>>>> Stage {STAGE_NAME2} completed. <<<<< \n")
    except Exception as e:
        logger.exception(e)
        raise e
    
    try:
        logger.info(f">>>>> Stage {STAGE_NAME3} started <<<<<")
        model_trainer = ModelTrainingPipeline()
        model_trainer.main()
        logger.info(f">>>>> Stage {STAGE_NAME3} completed. <<<<< \n")
    except Exception as e:
        logger.exception(e)
        raise e
    
    try:
        logger.info(f">>>>> Stage {STAGE_NAME4} started <<<<<")
        model_evaluator = ModelEvaluationPipeline()
        model_evaluator.main()
        logger.info(f">>>>> Stage {STAGE_NAME4} completed. <<<<< \n")
    except Exception as e:
        logger.exception(e)
        raise e
    
    try:
        logger.info(f">>>>> Stage {STAGE_NAME5} started <<<<<")
        model_trainer = ModelInferencePipeline()
        model_trainer.main()
        logger.info(f">>>>> Stage {STAGE_NAME5} completed. <<<<< \n")
    except Exception as e:
        logger.exception(e)
        raise e