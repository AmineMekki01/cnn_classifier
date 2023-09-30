# Brain MRI Multi-Class Classification Project

![image](https://github.com/AmineMekki01/cnn_classifier/assets/114087044/3023998b-ac9c-425c-87f4-877e09b1c514)

## Overview

This project aims to classify brain MRI images into one of four categories: three types of pathologies along with a no-tumor classification. By utilizing pre-trained models, the final layer is modified to accommodate the four classes essential for multi-class classification. The endeavor follows good practices of machine learning and programming, presenting an end-to-end project from data procurement from a GitHub repository to deployment. The structure of the project is meticulously organized to ensure clarity and detailed workflow in machine learning projects. Various tools and techniques such as configuration files, configuration manager, pipeline management, inference optimization (ONNX),   GitHub actions, and DVC for data versioning are implemented to streamline the process.


## Setup and Configuration
### Configurations
Configuration files are used to manage project settings. The primary configuration file is config.yaml located in the config directory. Moreover, a configuration manager is utilized to handle configurations throughout the project.

### GitHub Actions
GitHub Actions are employed to automate the workflow, ensuring consistency and reliability in the development process.

### DVC (Data Version Control)
DVC is used for data versioning, enabling tracking and versioning of datasets, making the project reproducible and shareable.

## Execution Pipeline
The project follows a structured pipeline divided into five stages:

1.Data Ingestion
2.Base Model Preparation
3.Training
4.Evaluation
5.Inference

Each stage is represented by scripts S1_data_ingestion.py, S2_prepare_base_model.py, S3_training.py, S4_evaluation.py, and S5_inferencer.py respectively under the src/cnnClassifier/pipeline directory.

## Web APP & UI
The project includes a basic UI to interact with the model. The UI files are located under the static and templates directories.
Created a streamlined FastAPI application for online inference, utilizing ONNX for optimization.

## Logging
Logs are maintained in the logs directory with log files named with the date of logging, aiding in debugging and tracking the project's progress.

## Conclusion
This project demonstrates a structured and well-organized approach to a machine learning project, right from data ingestion to deployment, following good programming and ML practices.
