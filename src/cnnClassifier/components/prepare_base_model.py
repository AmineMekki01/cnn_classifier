import torch
from torch import nn
from torchvision import models
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel(nn.Module):
    def __init__(self, config : PrepareBaseModelConfig):
        super(PrepareBaseModel, self).__init__()
        self.config = config
        self.num_classes = self.config.params_classes
        self.original_model = models.resnet18(pretrained=True)
        self.save_original_model(self.config.base_model_path)
        self.original_model = nn.Sequential(*list(self.original_model.children())[:-1])
        self.fc = nn.Linear(512, self.num_classes)
        self.save_modified_model(self.config.updated_base_model_path)

    def forward(self, x):
        x = self.original_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def save_original_model(self, path : Path):
        """
        This function saves the original model  
    
        Parameters
        ----------  
        path : Path
            path to save the original model
        
        Returns 
        -------
        None
        """
        torch.save(self.original_model, path)

    def save_modified_model(self, path : Path):
        """
        This function saves the modified model
        
        Parameters
        ----------
        path : Path
            path to save the modified model
        
        Returns 
        -------
        None
        """
        modified_model = nn.Sequential(
            self.original_model,
            self.fc
        )
        torch.save(modified_model, path)
    
    
