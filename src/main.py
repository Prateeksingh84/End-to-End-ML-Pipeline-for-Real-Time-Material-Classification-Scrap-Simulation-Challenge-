import sys
import os

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from simulation import simulate_real_time_loop
from data_preparation import prepare_dataset
from model_training import save_trained_model, train_and_evaluate_model
from inference_script import convert_model_to_onnx
import torch
from torchvision import models
import torch.nn as nn

if __name__ == '__main__':
    # Define paths
    data_dir = 'data/'
    trained_model_path = 'models/trained_model.pth'
    onnx_model_path = 'models/converted_model.onnx'
    output_csv = 'results/simulation_results.csv'
    
    # Check if the required folders exist, if not, create them
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 1. Prepare Dataset
    if os.path.exists(data_dir):
        train_loader, val_loader, class_names = prepare_dataset(data_dir)
        num_classes = len(class_names)
    else:
        print(f"Error: Dataset directory '{data_dir}' not found. Please add your data to this folder.")
        exit()

    # 2. Train and Convert Model
    print("Starting model training...")
    trained_model = train_and_evaluate_model(train_loader, val_loader, num_classes, class_names, model_name='resnet18')
    save_trained_model(trained_model, trained_model_path)
    
    try:
        # Load the trained model's state and convert it to ONNX
        trained_model_loaded = models.resnet18(weights=None)
        trained_model_loaded.fc = nn.Linear(trained_model_loaded.fc.in_features, num_classes)
        trained_model_loaded.load_state_dict(torch.load(trained_model_path))
        trained_model_loaded.eval()
        convert_model_to_onnx(trained_model_loaded, onnx_model_path)
    except Exception as e:
        print(f"Error converting model to ONNX: {e}")
        print("Please ensure you have run the data preparation and training steps successfully.")
        exit()

    # 3. Run the Simulation with the specified folder
    # Use the full path to the test_frames folder
    test_frames_folder = os.path.join(data_dir, 'test_frames')
    if not os.path.exists(test_frames_folder):
        print(f"Error: Test frames directory '{test_frames_folder}' not found. Please add images to this folder.")
    else:
        print(f"\nClassifying all images in: {test_frames_folder}")
        simulate_real_time_loop(test_frames_folder, onnx_model_path, class_names, output_csv)