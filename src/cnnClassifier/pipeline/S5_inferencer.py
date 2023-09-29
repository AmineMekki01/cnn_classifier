from cnnClassifier.constants import *
from cnnClassifier.components.inference import Inference
from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

STAGE_NAME = "Inference"

class ModelInferencePipeline:
    def __init__(self):
        pass
    
    def main(self):
        logger.info(f"Stage: {STAGE_NAME}")
        config_manager = ConfigurationManager()        
        testing_config = config_manager.get_inference_config()
        inference = Inference(config=testing_config)
        predictions = inference.predict_class()
        logger.info(predictions)
        logger.info("done")

            
if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
        data_ingestor = ModelInferencePipeline()
        data_ingestor.main()
        logger.info(f">>>>> Stage {STAGE_NAME} completed. <<<<< \n")
        
    except Exception as e:
        logger.exception(e)
        raise e
