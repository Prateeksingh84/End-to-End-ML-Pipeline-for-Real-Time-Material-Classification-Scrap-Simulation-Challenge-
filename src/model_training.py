import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import os

def train_and_evaluate_model(train_loader, val_loader, num_classes, class_names, model_name='resnet18', num_epochs=10):
    """
    Trains and evaluates a classification model using transfer learning.
    
    Args:
        train_loader (DataLoader): The DataLoader for the training data.
        val_loader (DataLoader): The DataLoader for the validation data.
        num_classes (int): The number of output classes.
        class_names (list): The list of all class names.
        model_name (str): The name of the CNN architecture to use ('resnet18' or 'mobilenet_v2').
        num_epochs (int): The number of training epochs.
        
    Returns:
        torch.nn.Module: The trained model.
    """
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Unsupported model architecture. Please use 'resnet18' or 'mobilenet_v2'.")

    # Use transfer learning by modifying the final layer
    for param in model.parameters():
        param.requires_grad = False
    
    if model_name == 'resnet18':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name == 'mobilenet_v2':
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        for param in model.classifier[1].parameters():
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}")

        # Evaluation with metrics: accuracy, precision, recall, confusion matrix
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Create a list of all possible class indices (e.g., [0, 1, 2, 3, 4])
        labels_to_show = list(range(num_classes))
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0, labels=labels_to_show)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0, labels=labels_to_show)
        conf_matrix = confusion_matrix(all_labels, all_preds, labels=labels_to_show)

        print(f"Validation Metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        print("Confusion Matrix:\n", conf_matrix)

    return model

def save_trained_model(model, file_path):
    """
    Saves the state dictionary of a trained PyTorch model to the models folder.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(model.state_dict(), file_path)
    print(f"Trained model saved successfully at {file_path}")

if __name__ == '__main__':
    # This block is for demonstration. In a full pipeline, a main.py script
    # would call these functions in sequence.
    from data_preparation import prepare_dataset
    from inference_script import convert_model_to_onnx

    # Define paths and parameters
    dataset_path = 'data/'
    trained_model_path = 'models/trained_model.pth'
    onnx_model_path = 'models/converted_model.onnx'

    if os.path.exists(dataset_path):
        train_loader, val_loader, class_names = prepare_dataset(dataset_path)
        num_classes = len(class_names)
        
        print("Starting model training...")
        trained_model = train_and_evaluate_model(train_loader, val_loader, num_classes, class_names, model_name='resnet18')
        
        save_trained_model(trained_model, trained_model_path)

        # Load the trained model's state and convert it to ONNX
        try:
            trained_model_loaded = models.resnet18(weights=None)
            trained_model_loaded.fc = nn.Linear(trained_model_loaded.fc.in_features, num_classes)
            trained_model_loaded.load_state_dict(torch.load(trained_model_path))
            trained_model_loaded.eval()
            convert_model_to_onnx(trained_model_loaded, onnx_model_path)
        except Exception as e:
            print(f"Error converting model to ONNX: {e}")
            print("Please ensure you have run the data preparation and training steps successfully.")

    else:
        print(f"Error: Dataset directory '{dataset_path}' not found.")