from torch import nn, optim
from cnnClassifier.constants import *
from cnnClassifier.components.training import Training
from cnnClassifier.components.prepare_callbacks import PrepareCallback
from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        logger.info(f"Stage: {STAGE_NAME}")
        config_manager = ConfigurationManager()
        prepare_callbacks_config = config_manager.get_prepare_callback_config()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()
        
        training_config = config_manager.get_training_config()
        
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train(callback_list=callback_list)
        
        logger.info(f"Model saved at {training_config.trained_model_path}")

if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
        data_ingestor = ModelTrainingPipeline()
        data_ingestor.main()
        logger.info(f">>>>> Stage {STAGE_NAME} completed. <<<<< \n")
        
    except Exception as e:
        logger.exception(e)
        raise e
