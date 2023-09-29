import torch
from torch import nn, optim
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from cnnClassifier.utils.common_functions import compute_metrics, save_json
from cnnClassifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config : TrainingConfig):
        """
        This function initializes the class
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    def get_model(self):
        """
        This function loads the model from the path provided in the config file
        """
        self.model = torch.load(self.config.updated_base_model_path)

        
    def train_valid_generator(self):
        """
        This function creates the train and validation generators
        """
        transform_list = [transforms.Resize(self.config.params_image_size[:-1]), 
                        transforms.ToTensor()]
        
        if self.config.params_is_augmentation:
            augmentations = [
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(self.config.params_image_size[:-1], scale=(0.8, 1.0), ratio=(0.75, 1.33), interpolation=2),
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
            
            transform_list = augmentations + transform_list
        
        train_transform = transforms.Compose(transform_list)
        
        full_dataset = ImageFolder(self.config.training_data, transform=train_transform)
        
        val_len = int(0.2 * len(full_dataset))
        train_len = len(full_dataset) - val_len

        self.train_dataset, self.valid_dataset = random_split(full_dataset, [train_len, val_len])
        
        self.train_dataset.class_to_idx = full_dataset.class_to_idx
        self.valid_dataset.class_to_idx = full_dataset.class_to_idx
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.params_batch_size, shuffle=True, num_workers=4)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.config.params_batch_size, shuffle=False, num_workers=4)

    def train_model(self, num_epochs : int, learning_rate : float):
        """
        This function trains the model
    
        Parameters
        ----------
        num_epochs : int
            Number of epochs to train the model 
        learning_rate : float   
            Learning rate for the optimizer 
        
        Returns
        -------
        None
        """
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        num_batches_train = len(self.train_loader)
        num_batches_valid = len(self.valid_loader)

        metrics={}
        
        for epoch in range(num_epochs):
            self.model.train()
            
            train_losses, train_accuracy, train_precision, train_recall, train_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
            valid_losses, valid_accuracy, valid_precision, valid_recall, valid_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
            
            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs or self.config.params_epochs}", unit="batch"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_losses += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                train_acc, train_prec, train_rec, train_f1_score  = compute_metrics(labels, predicted)
                
                train_accuracy += train_acc
                train_precision += train_prec
                train_recall += train_rec
                train_f1 += train_f1_score

            metrics[epoch] = {'train_loss': train_losses/num_batches_train, 'train_acc': train_accuracy/num_batches_train, 'train_prec': train_precision/num_batches_train, 'train_rec': train_recall/num_batches_train, 'train_f1': train_f1/num_batches_train}

            self.model.eval()
            with torch.no_grad():
                for images, labels in self.valid_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    valid_losses += loss.item()
                    
                    valid_acc, valid_prec, valid_rec, valid_f1_score = compute_metrics(labels, predicted)
                    
                    valid_accuracy += valid_acc
                    valid_precision += valid_prec
                    valid_recall += valid_rec
                    valid_f1 += valid_f1_score
            
            metrics[epoch] = {'valid_loss': valid_losses/num_batches_valid, 'valid_acc': valid_accuracy/num_batches_valid, 'valid_prec': valid_precision/num_batches_valid, 'valid_rec': valid_recall/num_batches_valid, 'valid_f1': valid_f1/num_batches_valid}
                
            print(f'Epoch {epoch+1}/{self.config.params_epochs} : Train Loss: {train_losses/num_batches_train}, Train Accuracy {train_accuracy/num_batches_train}, Train Precision {train_precision/num_batches_train}, Train Recall {train_recall/num_batches_train}, Train F1 Score {train_f1/num_batches_train}')

             
            print(f'Epoch {epoch+1}/{self.config.params_epochs} : Valid Loss: {valid_losses/num_batches_valid}, Valid Accuracy {valid_accuracy/num_batches_valid}, Valid Precision {valid_precision/num_batches_valid}, Valid Recall {valid_recall/num_batches_valid}, Valid F1 Score {valid_f1/num_batches_valid}')
            
        self.save_model(self.config.trained_model_path, self.model, self.train_dataset.class_to_idx)
        save_json(Path(self.config.score_path) / "metrics.json", metrics) 


    def save_model(self, path: Path, model: nn.Module, class_to_idx : dict):
        """
        This function saves the model to the path provided in the config file
        
        Parameters
        ----------
        path : Path
            Path to save the model
        model : nn.Module   
            Model to save
        class_to_idx : dict
            Class to index mapping
        
        Returns
        -------
        None
        """
        model.class_to_idx = class_to_idx  
        torch.save(model, path.with_suffix('.pth'))  

        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        onnx_path = path.with_suffix('.onnx')  
        
        torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=11, 
                          do_constant_folding=True, input_names=['input'], output_names=['output'])
