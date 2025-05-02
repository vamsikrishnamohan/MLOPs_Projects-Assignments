import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import precision_score, recall_score
import yaml
import os
import json
import logging
import numpy as np
from dvclive import Live
from logging_utilis import setup_logging

setup_logging("train.log")

class CNN(nn.Module):
    def __init__(self, conv_layers=3, conv_filters=32, kernel_size=3):
        super(CNN, self).__init__()
        layers = []
        in_channels = 3
        
        # Add convolutional layers
        for _ in range(conv_layers):
            layers += [
                nn.Conv2d(in_channels, conv_filters, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ]
            in_channels = conv_filters
            
        self.features = nn.Sequential(*layers)
        
        # Dynamically calculate the size of the feature map
        self.feature_size = self._calculate_feature_size()
        
        # Define the fully connected layer
        self.classifier = nn.Linear(self.feature_size, 10)

    def _calculate_feature_size(self):
        # Create a dummy input to calculate the size of the feature map
        dummy_input = torch.zeros(1, 3, 32, 32)  # Assuming input images are 32x32
        with torch.no_grad():
            dummy_output = self.features(dummy_input)
        return dummy_output.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the feature map
        return self.classifier(x)

def train_model(lr, conv_layers, conv_filters, kernel_size):
    try:
        logging.info("Starting training process...")
        
        # Load configuration
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        
        seed = int(params['seed'])
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        version = params['data']['version']
        
        # Initialize DVCLive
        live = Live("dvclive")

        # Data preparation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = datasets.ImageFolder(
            root=f"data/processed/{version}/train",  # Use version-specific train data
            transform=transform
        )
        
        val_dataset = datasets.ImageFolder(
            root=f"data/processed/{version}/val",  # Use version-specific val data
            transform=transform
        )
    
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNN(conv_layers=conv_layers, conv_filters=conv_filters, kernel_size=kernel_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        best_val_acc = 0
        for epoch in range(10):  # Fixed number of epochs
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            val_acc = correct / total
            val_precision = precision_score(all_labels, all_preds, average="macro")
            val_recall = recall_score(all_labels, all_preds, average="macro")
            
            # Log metrics with DVCLive
            live.log_metric("epoch", epoch + 1)
            live.log_metric("loss", running_loss / len(train_loader))
            live.log_metric("val_accuracy", val_acc)
            live.log_metric("val_precision", val_precision)
            live.log_metric("val_recall", val_recall)
            live.next_step()
            
            logging.info(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, "
                        f"Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, "
                        f"Val Recall: {val_recall:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), f"models/model_lr{lr}_layers{conv_layers}_filters{conv_filters}_kernel{kernel_size}.pth")
            
        logging.info(f"Best model saved with val acc: {best_val_acc:.4f}")

        # Save validation metrics
        os.makedirs("metrics", exist_ok=True)
        with open(f"metrics/val_accuracy_lr{lr}_layers{conv_layers}_filters{conv_filters}_kernel{kernel_size}.json", "w") as f:
            json.dump({"val_accuracy": best_val_acc}, f)
        
        logging.info("Training completed successfully.")
        return best_val_acc

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Load configuration
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    
    # Hyperparameter combinations
    lr_list = params['model']['lr']
    conv_layers_list = params['model']['conv_layers']
    conv_filters_list = params['model']['conv_filters']
    kernel_size_list = params['model']['kernel_size']
    
    # Perform grid search
    best_accuracy = 0
    best_hyperparams = {}
    
    for lr in lr_list:
        for conv_layers in conv_layers_list:
            for conv_filters in conv_filters_list:
                for kernel_size in kernel_size_list:
                    logging.info(f"Training with lr={lr}, conv_layers={conv_layers}, conv_filters={conv_filters}, kernel_size={kernel_size}")
                    val_accuracy = train_model(lr, conv_layers, conv_filters, kernel_size)
                    
                    # Track the best model
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_hyperparams = {
                            "lr": lr,
                            "conv_layers": conv_layers,
                            "conv_filters": conv_filters,
                            "kernel_size": kernel_size
                        }
    
    # Log the best hyperparameters
    logging.info(f"Best validation accuracy: {best_accuracy:.4f}")
    logging.info(f"Best hyperparameters: {best_hyperparams}")
    
    # Save the best hyperparameters
    with open("best_hyperparams.json", "w") as f:
        json.dump(best_hyperparams, f)