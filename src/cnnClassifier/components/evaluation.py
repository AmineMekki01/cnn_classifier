import torch
from torch import nn, optim
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from cnnClassifier.entity.config_entity import EvaluationConfig, TrainingConfig
from cnnClassifier.utils.common_functions import compute_metrics2, save_json
import os 

class Evaluation():
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.get_base_model()
        self.score = None
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Move the model to the device

    def get_base_model(self):
        """Load a saved model."""
        self.model = torch.load(self.config.path_to_model)

    def valid_generator(self):
        """Initialize the validation data loader."""
        # Only basic transforms for validation
        valid_transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]), 
            transforms.ToTensor()
        ])
        
        full_dataset = ImageFolder(self.config.training_data, transform=valid_transform)
        
        # one_percent_length = int(0.01 * len(full_dataset))
        # _, full_dataset = random_split(full_dataset, [len(full_dataset) - one_percent_length, one_percent_length])
        
        val_len = int(0.2 * len(full_dataset))
        self.valid_dataset, _ = random_split(full_dataset, [val_len, len(full_dataset) - val_len])
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.config.params_batch_size, shuffle=False, num_workers=4)
        
    def validate(self):
        """Evaluate the model on validation data."""
        self.model.eval()
        loss = 0.0
        acc = 0.0
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        
        metrics = {
                "validation precision" : [],
                "validation recall" : [],
                "validation f1 score" : [],
                "validation loss" : [],
                "validation accuracy" : []
        }
        
        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss += loss.item()
                print("input : " , inputs.shape)
                print("outputs : ", outputs.shape)
                valid_accuracy, valid_precision, valid_recall, valid_f1 = compute_metrics2(outputs, labels)
                
                acc += valid_accuracy
                precision += valid_precision
                recall += valid_recall
                f1_score += valid_f1
                
        precision /= len(self.valid_loader)
        recall /= len(self.valid_loader)
        f1_score /= len(self.valid_loader)
        acc /= len(self.valid_loader)
        loss /= len(self.valid_loader)

        metrics["validation precision"]= precision
        metrics["validation recall"]= recall
        metrics["validation f1 score"]= f1_score
        metrics["validation accuracy"]= acc
        metrics["validation loss"]= loss

        print(f"Train Accuracy: {acc:.4f} - Train Precision: {precision:.4f} - Train Recall: {recall:.4f} - Train F1: {f1_score:.4f} - Loss: {loss:.4f}")
        
        save_json(Path(self.config.score_path) / "metrics.json", metrics)

    def evaluation(self):
        """Main evaluation function to initialize data loaders and validate the model."""
        self.valid_generator()
        self.validate()
        
 