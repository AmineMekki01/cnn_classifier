import torch
from torch import nn, optim
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.utils.common_functions import compute_metrics1, save_json
import numpy as np

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
        
        metrics = {
            "epoch" : [],
            "train precision" : [],
            "train recall" : [],
            "validation precision" : [],
            "validation recall" : [],
            "train_f1" : [],
            "valid_f1" : [],
            "train_loss" : [],
            "train_accuracy" : [],
            "validation_loss" : [],
            "validation_accuracy" : []
        }
        
        for epoch in range(num_epochs or self.config.params_epochs):
            self.model.train()
            
            
            train_loss, train_accuracy, train_precision, train_recall, train_f1 = [], [], [], [], []
            valid_loss, valid_accuracy, valid_precision, valid_recall, valid_f1 = [], [], [], [], []
            
            for callback in callback_list:
                if hasattr(callback, "on_epoch_start"):
                    callback.on_epoch_start(epoch)

            # Training Loop with tqdm progress bar
            for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs or self.config.params_epochs}", unit="batch"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                tloss = self.criterion(outputs, labels)
                tloss.backward()
                self.optimizer.step()
                train_loss.append(tloss.item())
                train_acc, train_prec, train_rec, train_f1_score  = compute_metrics1(labels, outputs)
                train_accuracy.append(train_acc)
                train_precision.append(train_prec)
                train_recall.append(train_rec)
                train_f1.append(train_f1_score)
                
                for callback in callback_list:
                    if hasattr(callback, "on_batch_end"):
                        callback.on_batch_end()
            
            # Validation Loop
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                for inputs, labels in self.valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    vloss = self.criterion(outputs, labels)
                    
                    valid_loss.append(vloss.item())
                    valid_acc, valid_prec, valid_rec, valid_f1_score = compute_metrics1(labels, outputs)

                    valid_accuracy.append(valid_acc)
                    valid_precision.append(valid_prec)
                    valid_recall.append(valid_rec)
                    valid_f1.append(valid_f1_score)

            
            print(f"Train Accuracy: {np.mean(train_accuracy):.4f} - Train Precision: {np.mean(train_precision):.4f} - Train Recall: {np.mean(train_recall):.4f} - Train F1: {np.mean(train_f1):.4f} - Train Loss: {np.mean(train_loss):.4f}")
            print(f"Validation Accuracy: {np.mean(valid_accuracy):.4f} - Validation Precision: {np.mean(valid_precision):.4f} - Validation Recall: {np.mean(valid_recall):.4f} - Validation F1: {np.mean(valid_f1):.4f} - Validation Loss: {np.mean(valid_loss):.4f}\n")

            metrics["epoch"].append(epoch)
            metrics["train precision"].append(np.mean(train_precision))
            metrics["train recall"].append(np.mean(train_recall))
            metrics["validation precision"].append(np.mean(valid_precision))
            metrics["validation recall"].append(np.mean(valid_recall))
            metrics["train_f1"].append(np.mean(train_f1))
            metrics["valid_f1"].append(np.mean(valid_f1))
            metrics["train_loss"].append(np.mean(train_loss))
            metrics["train_accuracy"].append(np.mean(train_accuracy))
            metrics["validation_loss"].append(np.mean(valid_loss))
            metrics["validation_accuracy"].append(np.mean(valid_accuracy))

            for callback in callback_list:
                if hasattr(callback, "on_epoch_end"):
                    callback.on_epoch_end(epoch)
            
        self.save_model(self.config.trained_model_path, self.model)
        save_json(Path(self.config.score_path) / "metrics.json", metrics)

