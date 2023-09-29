import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from cnnClassifier.entity.config_entity import InferenceConfig
from PIL import Image
import io
import onnx
import onnxruntime as ort
import numpy as np
class Inference:
    def __init__(self, config : InferenceConfig):
        """
        This function initializes the class
    
        Parameters
        ----------
        config : InferenceConfig
            config file 
        
        Returns 
        -------
        None
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_files = [f for f in os.listdir(self.config.testing_data) if os.path.isfile(os.path.join(self.config.testing_data, f))]
        self.transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size),
            transforms.ToTensor()
        ])
        self.idx_to_class = {
            0: 'glioma', 
            1: 'meningioma', 
            2: 'notumor', 
            3: 'pituitary', 
            }
        

    def predict(self, input_tensor : torch.Tensor) -> torch.Tensor:
        """
        This function predicts the class of the input image
        
        Parameters
        ----------
        input_tensor : torch.Tensor
            input tensor    
    
        Returns
        -------
        output : torch.Tensor
            output tensor   
        """
        ort_session = ort.InferenceSession(self.config.path_to_model, providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
        input_name = ort_session.get_inputs()[0].name
        try:
            ort_inputs = {input_name: np.squeeze(np.expand_dims(input_tensor.cpu().numpy(), 0), axis=0)}
            ort_outs = ort_session.run(None, ort_inputs)  
        except Exception as e:
            ort_inputs = {input_name: np.expand_dims(input_tensor.cpu().numpy(), 0)}  
            ort_outs = ort_session.run(None, ort_inputs)  
        return torch.Tensor(ort_outs[0])
    
    def predict_class(self) -> list:
        """
        This function predicts the class of the input image 
    
        Returns
        -------
        class_names : list
            list of class names 
        """
        loader = DataLoader(self, batch_size=self.config.params_batch_size, shuffle=False)

        predictions = []
        for i in range(len(self)):
            inputs = self.__getitem__(i)
            output = self.predict(inputs)
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.cpu())
        class_names = [self.idx_to_class[pred.item()] for pred in predictions] 
        return class_names
    
    def predict_single_image(self, image_input) -> str:
        """
        This function predicts the class of a single input image
        
        Parameters
        ----------
        image_input : str or PIL.Image.Image
            Either the file path of the image or a PIL Image object
        
        Returns
        -------
        class_name : str
            Name of the predicted class
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise TypeError("Input should be either a file path or a PIL Image object")
        
        input_tensor = self.transform(image)
        input_tensor = input_tensor.unsqueeze(0)
        output = self.predict(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        class_name = self.idx_to_class[predicted_idx.item()]
        return class_name
    
    def __len__(self) -> int:
        """
        This function returns the length of the dataset
        
        Returns
        ------- 
        len : int
            length of the dataset
        """
        return len(self.image_files)

    def __getitem__(self, idx : int) -> torch.Tensor:
        """
        This function returns the image at the given index
        
        Parameters  
        ----------  
        idx : int
            index of the image  
        
        Returns
        -------
        image : torch.Tensor
            image tensor
        """
        img_name = os.path.join(self.config.testing_data, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
