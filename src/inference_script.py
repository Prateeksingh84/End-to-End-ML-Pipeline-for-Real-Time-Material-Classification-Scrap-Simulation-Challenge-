import torch
import onnxruntime
from torchvision import transforms
from PIL import Image
import numpy as np
import os

def convert_model_to_onnx(model, onnx_path):
    """
    Converts a trained PyTorch model to the ONNX format.
    
    Args:
        model (torch.nn.Module): The trained PyTorch model.
        onnx_path (str): The desired path to save the ONNX file.
    """
    # Create a dummy input tensor that matches the expected input size
    # for a single image (batch size of 1).
    dummy_input = torch.randn(1, 3, 224, 224) 
    
    # Export the model to ONNX format
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        input_names=['input'], 
        output_names=['output'],
    )
    print(f"Model successfully converted to ONNX and saved at {onnx_path}")

class InferenceEngine:
    def __init__(self, model_path, class_names):
        """
        Initializes the inference engine with an ONNX model and class names.
        
        Args:
            model_path (str): Path to the ONNX model file.
            class_names (list): List of class names corresponding to model output.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found at: {model_path}")
            
        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.class_names = class_names
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classify_image(self, image_path):
        """
        Takes one image/frame and outputs the predicted class and confidence.
        
        Args:
            image_path (str): Path to the image file to classify.
            
        Returns:
            tuple: (predicted_class_name, confidence_score)
        """
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None, 0.0
            
        # Preprocess the image and add a batch dimension
        input_tensor = self.preprocess(image).unsqueeze(0)
        
        # Run inference using ONNX Runtime
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_tensor.numpy()}
        ort_outs = self.ort_session.run(None, ort_inputs)
        
        # Process the output to get the predicted class and confidence
        output = torch.from_numpy(ort_outs[0])
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class_idx = torch.max(probabilities, 1)
        
        predicted_class_name = self.class_names[predicted_class_idx.item()]
        
        return predicted_class_name, confidence.item()

# Example usage (for demonstration only):
if __name__ == '__main__':
    # You would typically run model_training.py and get the ONNX model from there.
    # For this example, we'll assume the model and a sample image exist.
    
    # Placeholder for model and class names
    onnx_path = 'models/converted_model.onnx'
    class_names = ['e-waste', 'fabric', 'metal', 'paper', 'plastic']
    sample_image_path = 'data/test_frames/sample_image.jpg'
    
    # Initialize the inference engine
    try:
        inference_engine = InferenceEngine(onnx_path, class_names)
        
        # Classify a sample image
        pred_class, confidence = inference_engine.classify_image(sample_image_path)
        
        if pred_class is not None:
            print(f"Predicted Class: {pred_class}")
            print(f"Confidence: {confidence:.4f}")
            
    except FileNotFoundError as e:
        print(f"An error occurred: {e}")