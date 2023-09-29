from cnnClassifier.constants import *
from cnnClassifier.components.evaluation import Evaluation
from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager

STAGE_NAME = "Evaluation"

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        logger.info(f"Stage: {STAGE_NAME}")
        config_manager = ConfigurationManager()
        evaluation_config = config_manager.get_evaluation_config()
        evaluation = Evaluation(config=evaluation_config)
        evaluation.evaluation()

if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
        evaluator = ModelEvaluationPipeline()
        evaluator.main()
        logger.info(f">>>>> Stage {STAGE_NAME} completed. <<<<< \n")
        
    except Exception as e:
        logger.exception(e)
        raise e
