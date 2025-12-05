# model_utils.py
import os
import sys
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from typing import Tuple

# add project folder (where model.py lives) to sys.path if not already
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if os.path.isdir(PROJECT_ROOT) and PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
else:
    alt = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if alt not in sys.path:
        sys.path.insert(0, alt)

# Paths (edit if needed)
CKPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best.pth"))
ONNX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "brain_tumor_model.onnx"))

DEFAULT_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

IMG_SIZE = 224
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ---------- model loading (robust) ----------
def load_torch_model(ckpt_path=CKPT_PATH, device=None, use_qnn=True):
    import numpy as _np
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = None
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
        except Exception as e_inner:
            try:
                torch.serialization.add_safe_globals([_np._core.multiarray.scalar])
            except Exception:
                try:
                    ctx = torch.serialization.safe_globals([_np._core.multiarray.scalar])
                    ctx.__enter__()
                    ckpt = torch.load(ckpt_path, map_location=device)
                    ctx.__exit__(None, None, None)
                except Exception:
                    raise e_inner
    except Exception as e:
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
        except Exception:
            raise e

    if ckpt is None:
        raise RuntimeError("Failed to load checkpoint.")

    classes = ckpt.get('classes', DEFAULT_CLASSES)
    from model import create_model
    # Hybrid model.py accepts use_qnn
    model = create_model(num_classes=len(classes), use_qnn=use_qnn)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model, classes, device

def preprocess_pil(img_pil: Image.Image) -> torch.Tensor:
    x = transform(img_pil)
    return x.unsqueeze(0)

def predict_torch(img_pil: Image.Image, model, classes, device):
    x = preprocess_pil(img_pil).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    return {
        "pred_class": classes[idx],
        "pred_idx": idx,
        "confidence": float(probs[idx]),
        "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))}
    }

# ---------- ONNX (optional) ----------
def load_onnx_model(onnx_path=ONNX_PATH):
    import onnxruntime as ort
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    classes = DEFAULT_CLASSES
    return sess, classes

def predict_onnx(img_pil: Image.Image, sess, classes):
    img = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2,0,1).astype(np.float32)
    img = np.expand_dims(img, 0)
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: img})[0]
    probs = softmax(out[0])
    idx = int(np.argmax(probs))
    return {
        "pred_class": classes[idx],
        "pred_idx": idx,
        "confidence": float(probs[idx]),
        "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))}
    }

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)

# ---------- Grad-CAM utilities ----------
import torch.nn as nn

def find_conv_modules(model: torch.nn.Module):
    """
    Return a list of (name, module) for all Conv2d modules in the model in traversal order.
    """
    convs = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            convs.append((name, mod))
    return convs

def compute_gradcam(model: torch.nn.Module, device, pil_img: Image.Image, target_class: int = None):
    """
    Standard Grad-CAM implementation:
    - Uses forward hook to get activations
    - Uses backward hook to get gradients w.r.t activations
    """
    model.eval()
    
    # Find the last convolutional layer
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module
    
    if target_layer is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM")
    
    # Prepare input
    x = preprocess_pil(pil_img).to(device)
    # We don't need x.requires_grad for the input image itself for Grad-CAM, 
    # but we need to ensure the model parameters require grad if we were training, 
    # but here we are in eval mode. 
    # However, to get gradients back to the layer, we need to ensure the graph is built.
    # Since we are in inference, we usually use torch.set_grad_enabled(True) context or just don't use no_grad.
    
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # Register hooks
    f_handle = target_layer.register_forward_hook(forward_hook)
    b_handle = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        # Forward pass
        # We need gradients, so we must NOT use torch.no_grad()
        # and we might need to ensure the input requires grad if the model was fully frozen?
        # Usually for inference models, params have requires_grad=True unless explicitly frozen.
        # If they are frozen, we can't get gradients w.r.t weights, but we can get w.r.t activations 
        # IF we ensure the tensor requires grad? No, intermediate tensors require grad if input or params do.
        # Let's assume params have requires_grad=True. If not, we might need to set them.
        
        output = model(x)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero grads
        model.zero_grad()
        
        # Target for backprop
        score = output[0, target_class]
        
        # Backward pass
        score.backward()
        
        # Generate heatmap
        if gradients is not None and activations is not None:
            # Global Average Pooling of gradients
            weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
            
            # Weighted combination of activations
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            
            # ReLU
            cam = torch.relu(cam)
            
            # Normalize
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            
            heatmap = cam.squeeze().detach().cpu().numpy()
        else:
            heatmap = np.zeros((7, 7))
            
        # Resize and overlay
        orig = np.array(pil_img.convert("RGB"))
        heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlay = (0.4 * colored_heatmap.astype(float) + 0.6 * orig.astype(float)).astype(np.uint8)
        
        return overlay, heatmap_resized
        
    finally:
        f_handle.remove()
        b_handle.remove()

# ---------- helper: tumor presence ----------
def tumor_present(pred_result):
    label = pred_result['pred_class'].lower()
    if label == 'notumor' or label == 'no tumor' or 'no' in label:
        return False
    return True
