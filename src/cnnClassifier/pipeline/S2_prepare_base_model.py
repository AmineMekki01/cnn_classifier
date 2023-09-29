from cnnClassifier.constants import *
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager
import torch



STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        logger.info(f"Stage: {STAGE_NAME}")
        config_manager = ConfigurationManager()
        config = config_manager.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config)
        torch.save(prepare_base_model, config.updated_base_model_path)
      
        logger.info(f"Updated base model saved at {config.updated_base_model_path}")

if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
        prepare_base_model = PrepareBaseModelTrainingPipeline()
        prepare_base_model.main()
        logger.info(f">>>>> Stage {STAGE_NAME} completed. <<<<< \n")
        
    except Exception as e:
        logger.exception(e)
        raise e