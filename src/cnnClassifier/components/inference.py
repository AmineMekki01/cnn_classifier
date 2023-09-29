import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cnnClassifier.entity.config_entity import InferenceConfig
from PIL import Image

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
        self.get_base_model()
        self.image_files = [f for f in os.listdir(self.config.testing_data) if os.path.isfile(os.path.join(self.config.testing_data, f))]
        self.transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size),
            transforms.ToTensor()
        ])

    def get_base_model(self):
        """
        This function loads the model from the path provided in the config file
        """
        self.model = torch.load(self.config.path_to_model)
        self.model.to(self.device)
        self.model.eval()
        self.class_to_idx = self.model.class_to_idx  
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}  

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
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
        return output
    
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
        for inputs in loader:
            output = self.predict(inputs)
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.cpu())
        class_names = [self.idx_to_class[pred.item()] for pred in predictions]  # Use idx_to_class for lookup
        return class_names
    
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
