import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from cnnClassifier.entity.config_entity import InferenceConfig
from PIL import Image

class Inference:
    def __init__(self, config = InferenceConfig):
        """
        Initialize the inference class.
        
        Parameters:
        - model_path (str): Path to the saved model.
        - device (torch.device): Device to run the inference on.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_base_model()
        
    def get_base_model(self):
        self.model = torch.load(self.config.path_to_model)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, input_tensor):
        """
        Make a prediction using the model.
        
        Parameters:
        - input_tensor (torch.Tensor): Tensor representing the input data.
        
        Returns:
        - output (torch.Tensor): Model's prediction.
        """
        
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
        return output

    def predict_from_path(self):
        """
        Load data from a path and make predictions.
        
        Parameters:
        - data_path (str): Path to the data.
        - transform (transforms.Compose): Image transformations.
        
        Returns:
        - predictions (list): List of predictions.
        """
        transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]), 
            transforms.ToTensor()
        ])
        
        dataset = ImageFolder(self.config.testing_data, transform=transform)
        loader = DataLoader(dataset, batch_size=self.config.params_batch_size, shuffle=False)
        
        predictions = []
        for inputs, _ in loader:
            output = self.predict(inputs)
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.cpu())
        return predictions


class CustomImageDataset(Dataset):
    def __init__(self, config=InferenceConfig):
        """
        Args:
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.config = config
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_base_model()
        self.image_files = [f for f in os.listdir(self.config.testing_data) if os.path.isfile(os.path.join(self.config.testing_data, f))]
        self.transform = transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.ToTensor()
        ])
        self.idx_to_class = {
            0: "glioma",
            1: "meningioma",
            2: "notumor",
            3: "pituitary"
        }

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.config.testing_data, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    
    def get_base_model(self):
        self.model = torch.load(self.config.path_to_model)
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, input_tensor):
        
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
        return output
    
    def predict_class(self):
        loader = DataLoader(self, batch_size=self.config.params_batch_size, shuffle=False)
        predictions = []
        for inputs in loader:
            output = self.predict(inputs)
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.cpu())
        
        class_names = [self.idx_to_class[pred.item()] for pred in predictions]
        return class_names  
    
    



