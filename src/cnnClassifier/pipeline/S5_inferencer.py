from torch import nn, optim
from cnnClassifier.constants import *
from cnnClassifier.components.inference import CustomImageDataset
from cnnClassifier.components.prepare_callbacks import PrepareCallback
from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager

STAGE_NAME = "Inference"

class ModelInferencePipeline:
    def __init__(self):
        pass
    
    def main(self):
        logger.info(f"Stage: {STAGE_NAME}")
        config_manager = ConfigurationManager()        
        testing_config = config_manager.get_inference_config()
        
        inference = CustomImageDataset(config=testing_config)
        
        predictions = inference.predict_class()
        print(predictions)
        logger.info("done")
        

