import os
import sys
import torch
from PIL import Image
import numpy as np
import traceback

# Add current directory to path
sys.path.append(os.getcwd())

LOG_FILE = "verification_log.txt"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    print(msg)

# Clear log file
with open(LOG_FILE, "w") as f:
    f.write("Starting verification...\n")

try:
    log("Importing model_utils_hybrid...")
    from model_utils_hybrid import load_torch_model, predict_torch, compute_gradcam
    log("Successfully imported model_utils_hybrid")
except ImportError as e:
    log(f"Failed to import model_utils_hybrid: {e}")
    sys.exit(1)
except Exception as e:
    log(f"Error during import: {e}")
    log(traceback.format_exc())
    sys.exit(1)

def test_gradcam():
    log("Testing Grad-CAM...")
    
    # Create a dummy image
    img = Image.new('RGB', (224, 224), color = 'red')
    
    try:
        # Load model
        log("Loading model...")
        model, classes, device = load_torch_model()
        log(f"Model loaded. Device: {device}")
        log(f"Classes: {classes}")
        
        # Predict
        log("Predicting...")
        res = predict_torch(img, model, classes, device)
        log(f"Prediction: {res}")
        
        # Compute Grad-CAM
        log("Computing Grad-CAM...")
        overlay, heatmap = compute_gradcam(model, device, img, target_class=res['pred_idx'])
        
        log(f"Grad-CAM computed. Overlay shape: {overlay.shape}, Heatmap shape: {heatmap.shape}")
        
        if heatmap.max() == 0 and heatmap.min() == 0:
            log("WARNING: Heatmap is all zeros.")
        else:
            log("Heatmap contains non-zero values.")
            
        log("Verification SUCCESS")
        
    except Exception as e:
        log(f"Verification FAILED: {e}")
        log(traceback.format_exc())

if __name__ == "__main__":
    test_gradcam()
