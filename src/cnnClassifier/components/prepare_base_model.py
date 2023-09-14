import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchsummary import summary
from pathlib import Path
from cnnClassifier.constants import *
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        # Note: torchvision has pre-trained weights only for include_top=True. So if you need exclude top, you'll need to handle that differently.
        self.model = models.vgg19(pretrained=(self.config.params_weights == "imagenet"))

        if not self.config.params_include_top:
            # Removing the classifier part
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.save_model(path=self.config.base_model_path, model=self.model)

    def _prepare_full_model(self, model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        elif (freeze_all is not None) and (freeze_all > 0):
            ct = 0
            for child in model.children():
                ct += 1
                if ct < freeze_till:
                    for param in child.parameters():
                        param.requires_grad = False

        # If include_top is False
        if not self.config.params_include_top:
            # Adding new layers
            model = nn.Sequential(
                model,
                nn.AdaptiveAvgPool2d((1, 1)),   # Adding Adaptive Average Pooling
                nn.Flatten(),
                nn.Linear(512, classes)  # Adjusted from 512 * 7 * 7 to just 512
            )
        else:
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, classes)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        return model, optimizer, criterion

    def update_base_model(self):
        self.full_model, optimizer, criterion = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model:nn.Module):
        torch.save(model, path)
    
    def get_summary(self, input_size):
        return summary(self.full_model, input_size)
