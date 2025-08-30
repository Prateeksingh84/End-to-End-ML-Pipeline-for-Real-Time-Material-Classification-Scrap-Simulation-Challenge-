import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import os

def prepare_dataset(data_dir, img_size=(224, 224), batch_size=32):
    """
    Loads, preprocesses, and augments the image dataset for training readiness.
    """
    train_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, class_names

if __name__ == '__main__':
    # This block is for testing the function directly
    dataset_path = 'data/'
    if os.path.exists(dataset_path):
        train_loader, val_loader, class_names = prepare_dataset(dataset_path)
        print(f"Dataset classes: {class_names}")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
    else:
        print(f"Error: Dataset directory '{dataset_path}' not found. Please place your dataset in a folder named 'data/'.")