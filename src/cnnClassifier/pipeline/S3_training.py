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
        training_config = config_manager.get_training_config()
        training = Training(config=training_config)
        training.get_model()
        training.train_valid_generator()
        training.train_model(training_config.params_epochs, training_config.params_learning_rate)
        logger.info(f"Model saved at {training_config.trained_model_path}")

if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
        model_trainer = ModelTrainingPipeline()
        model_trainer.main()
        logger.info(f">>>>> Stage {STAGE_NAME} completed. <<<<< \n")
        
    except Exception as e:
        logger.exception(e)
        raise e