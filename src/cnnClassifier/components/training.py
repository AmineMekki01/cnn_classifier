import torch
from torch import nn, optim
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.utils.common_functions import accuracy


class Training:
    def __init__(self, config = TrainingConfig):
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_base_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def get_base_model(self):
        self.model = torch.load(self.config.updated_base_model_path)

    def train_valid_generator(self):
        transform_list = [transforms.Resize(self.config.params_image_size[:-1]), 
                          transforms.ToTensor()]
        
        if self.config.params_is_augmentation:
            augmentations = [
                transforms.RandomRotation(40),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(self.config.params_image_size[:-1], scale=(0.8, 1.0),
                                             ratio=(0.75, 1.33), interpolation=2),
                transforms.RandomAffine(degrees=0, shear=20),
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))
            ]
            transform_list = augmentations + transform_list
        
        train_transform = transforms.Compose(transform_list)
        
        full_dataset = ImageFolder(self.config.training_data, transform=train_transform)
        
        val_len = int(0.2 * len(full_dataset))
        train_len = len(full_dataset) - val_len

        self.train_dataset, self.valid_dataset = random_split(full_dataset, [train_len, val_len])
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.params_batch_size, shuffle=True, num_workers=4)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.config.params_batch_size, shuffle=False, num_workers=4)

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model, path)

    def train(self, callback_list=[], num_epochs=None):
        self.model.to(self.device)
        
        for epoch in range(num_epochs or self.config.params_epochs):
            self.model.train()
            
            train_loss = 0.0
            train_acc = 0.0
            valid_loss = 0.0
            valid_acc = 0.0
            
            for callback in callback_list:
                if hasattr(callback, "on_epoch_start"):
                    callback.on_epoch_start(epoch)

            # Training Loop with tqdm progress bar
            for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs or self.config.params_epochs}", unit="batch"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_acc += accuracy(outputs, labels)
                
                for callback in callback_list:
                    if hasattr(callback, "on_batch_end"):
                        callback.on_batch_end()
            
            # Validation Loop
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                for inputs, labels in self.valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    valid_loss += loss.item()
                    valid_acc += accuracy(outputs, labels)

            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader)
            valid_loss /= len(self.valid_loader)
            valid_acc /= len(self.valid_loader)

            print(f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_acc:.4f}")
            print(f"Validation Loss: {valid_loss:.4f} - Validation Accuracy: {valid_acc:.4f}\n")

            for callback in callback_list:
                if hasattr(callback, "on_epoch_end"):
                    callback.on_epoch_end(epoch)
            
        self.save_model(self.config.trained_model_path, self.model)

