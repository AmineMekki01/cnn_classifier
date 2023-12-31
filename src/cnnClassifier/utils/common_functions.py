"""
Script_name : common_functions.py
Author_name : Amine MEKKI

Description:
In this file i will put the common used functions so as i dont repeat my self.
"""

# importing libraries

import numpy as np
import os 
from cnnClassifier import logger
import yaml
import json 
import base64
import torch
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc

@ensure_annotations
def read_yaml(path_to_yaml : Path) -> ConfigBox:
    """
    This function reads a yaml file and returns a ConfigBox object. 

    Parameters
    ----------
    path_to_yaml : Path
        path to yaml file.

    Raises:
        ValueError: if yaml file is empty.
        e: if any other error occurs.
    
    Returns:
    -------
        ConfigBox : ConfigBox object.
    """
    try: 
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Yaml file : {os.path.normpath(path_to_yaml)} loaded successfully.")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty.")
    except Exception as e:
        raise e  

@ensure_annotations
def create_directories(path_to_directories : list, verbose : bool = True):
    """
    This function creates directories if they dont exist.
    
    Parameters
    ----------
    path_to_directories : list
        list of paths to directories.   
    verbose : bool, optional
        if True, print the created directories, by default True 
    
    Returns
    -------
    None
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at : {os.path.normpath(path)}")
            
@ensure_annotations
def save_json(path : Path, data :dict):
    """
    This function saves a json file.    

    Parameters
    ----------
    path : Path
        path to save the json file. 
    
    data : dict 
        data to save in the json file.
    
    Returns
    -------
    None
    """

    def convert_tensors_to_lists(item : dict):
        """
        This function converts tensors to lists.
    
        Parameters  
        ----------  
        item : dict, torch.Tensor, list
            item to convert.    
        
        Returns 
        ------- 
        dict, list, torch.Tensor
            converted item. 
            
        """
        if isinstance(item, dict):
            return {key: convert_tensors_to_lists(value) for key, value in item.items()}
        elif isinstance(item, torch.Tensor):
            return item.cpu().numpy().tolist()
        elif isinstance(item, list):
            return [convert_tensors_to_lists(i) for i in item]
        else:
            return item

    data = convert_tensors_to_lists(data)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at : {os.path.normpath(path)}")
    
    
@ensure_annotations
def load_json(path : Path) -> ConfigBox:
    """
    This function loads a json file.
    
    Parameters
    ----------
    path : Path
        path to json file.  
    
    Returns 
    -------
    ConfigBox
        ConfigBox object.   
    """
    with open(path) as f:
        content = json.load(f)
        
    logger.info(f"json file loaded successfully from : {os.path.normpath(path)}")
    return ConfigBox(content)    

@ensure_annotations
def get_size(path : Path) -> str:
    """
    get size in KB
    
    Parameters
    ----------
    path : Path
        path to file.   
    
    Returns 
    ------- 
    str
        size in KB.
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB."

def decodeImage(img_string : str, file_name : str):
    """
    This function decodes an image from base64 string and saves it.

    Parameters 
    ----------
    img_string : str

    file_name : str 

    Returns 
    ------- 
    None    
    """
    img_data = base64.b64decode(img_string)
    with open(file_name, "wb") as f:
        f.write(img_data)
        f.close()
        
        
def encodeImageIntoBase64(cropped_image_path : str) -> str:
    """
    This function encodes cropped images into base 64 strings for sending them through API calls.   
    
    Parameters  
    ----------
    cropped_image_path : str
        path to cropped image.
    
    Returns 
    -------
    str
        base64 string.      
    """
    with open(cropped_image_path, "rb") as f:
        img_string = base64.b64encode(f.read())
        f.close()
    return img_string #.decode('utf-8')

def compute_metrics(y_true : torch.Tensor, y_pred : torch.Tensor) -> float:
    """
    This function computes metrics for multiclass classification.   
    
    Parameters
    ----------
    y_true : torch.Tensor
        true labels.
    
    y_pred : torch.Tensor   
        predicted labels.
    
    Returns 
    ------- 
    float
        accuracy.   
    
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    precision = precision_score(y_true, y_pred, zero_division=1, average='weighted')
    recall = recall_score(y_true, y_pred, zero_division=1, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    acc = accuracy_score(y_true, y_pred)

    return acc, precision, recall, f1

def compute_metrics2(y_pred, y_true):
    if y_pred.dim() > 1: 
        
        y_pred = torch.argmax(y_pred, dim=1)
    y_pred = y_pred.cpu().numpy()  
    y_true = y_true.cpu().numpy()

    precision = precision_score(y_true, y_pred, zero_division=1, average='weighted')
    recall = recall_score(y_true, y_pred, zero_division=1, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)

    return acc, precision, recall, f1

def get_class_info_from_data(data_dir):
    classes = sorted(os.listdir(data_dir))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    return class_to_idx, classes