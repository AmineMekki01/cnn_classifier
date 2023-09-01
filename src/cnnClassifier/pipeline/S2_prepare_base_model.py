from cnnClassifier.constants import *
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager




STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        logger.info(f"Stage: {STAGE_NAME}")
        config_manager = ConfigurationManager()
        config = config_manager.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()
        logger.info(f"Base model prepared and saved at {config.base_model_path}")
        logger.info(f"Updated base model saved at {config.updated_base_model_path}")

# # testing
# if __name__ == "__main__":
#     pipeline = PrepareBaseModelTrainingPipeline()
#     pipeline.main()