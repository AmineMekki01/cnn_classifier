import torch
from torch import nn, optim
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from cnnClassifier.entity.config_entity import EvaluationConfig, TrainingConfig
from cnnClassifier.utils.common_functions import accuracy, save_json


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
        
        # Assuming the validation data is 20% of the full dataset
        val_len = int(0.2 * len(full_dataset))
        self.valid_dataset, _ = random_split(full_dataset, [val_len, len(full_dataset) - val_len])
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.config.params_batch_size, shuffle=False, num_workers=4)
        
    def validate(self):
        """Evaluate the model on validation data."""
        self.model.eval()
        valid_loss = 0.0
        valid_acc = 0.0
        
        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                valid_loss += loss.item()
                valid_acc += accuracy(outputs, labels)
            
        valid_loss /= len(self.valid_loader)
        valid_acc /= len(self.valid_loader)
        print(f"Validation Loss: {valid_loss:.4f} - Validation Accuracy: {valid_acc:.4f}\n")
        
        return (valid_loss, valid_acc)
    
    def evaluation(self):
        """Main evaluation function to initialize data loaders and validate the model."""
        self.valid_generator()
        self.score = self.validate()
        
    def save_score(self):
        """Save the evaluation scores as a JSON file."""
        if self.score:
            scores = {"Loss": self.score[0], "Accuracy": self.score[1]}
            save_json(self.config.score_path, scores)
        else:
            print("Scores have not been computed. Run the evaluation first.")
