import os
import csv
import time
from inference_script import InferenceEngine

def simulate_real_time_loop(image_folder, model_path, class_names, output_csv, low_confidence_threshold=0.7):
    """
    Builds a dummy conveyor simulation to classify images at intervals.
    Includes bonus features for manual override and active learning.
    
    Args:
        image_folder (str): Directory containing images to process.
        model_path (str): Path to the converted ONNX model.
        class_names (list): List of class names.
        output_csv (str): Path to the output CSV file.
        low_confidence_threshold (float): Confidence threshold for flagging low-confidence predictions.
    """
    inference_engine = InferenceEngine(model_path, class_names)
    retraining_queue = [] # This list will act as our active learning queue

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_id', 'image_name', 'predicted_class', 'confidence', 'low_confidence_flag', 'retraining_flag'])

    all_items = sorted(os.listdir(image_folder))
    image_files = [item for item in all_items if os.path.isfile(os.path.join(image_folder, item))]
    
    print("Starting real-time scrap classification simulation...")
    
    for frame_id, image_name in enumerate(image_files):
        image_path = os.path.join(image_folder, image_name)
        
        # Simulate capturing a frame at an interval
        time.sleep(0.5)

        pred_class, confidence = inference_engine.classify_image(image_path)
        
        # Skip if classification failed (e.g., file was not an image)
        if pred_class is None:
            continue

        # Check for low confidence (Manual Override Logic)
        low_conf_flag = confidence < low_confidence_threshold
        retraining_flag = False

        if low_conf_flag:
            # Simulate an "active learning" decision to add the image to a retraining queue
            retraining_queue.append(image_path)
            retraining_flag = True
        
        # Log output to console
        log_message = f"Frame {frame_id}: {image_name} -> Class: {pred_class}, Confidence: {confidence:.2f}"
        if low_conf_flag:
            log_message += " (LOW CONFIDENCE)"
            print(f"  >>> Manual Override: Adding {image_name} to retraining queue. <<<")
        print(log_message)

        # Store in a result CSV
        with open(output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([frame_id, image_name, pred_class, confidence, low_conf_flag, retraining_flag])
    
    print("\nSimulation complete. Results saved to", output_csv)
    print("\n--- Active Learning Queue ---")
    if retraining_queue:
        print(f"The following {len(retraining_queue)} images have been flagged for retraining:")
        for item in retraining_queue:
            print(f"- {item}")
    else:
        print("No images were flagged for retraining.")