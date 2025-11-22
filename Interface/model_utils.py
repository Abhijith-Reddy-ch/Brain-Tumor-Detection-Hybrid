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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "BrainTumorCNN"))
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
def load_torch_model(ckpt_path=CKPT_PATH, device=None):
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
    model = create_model(num_classes=len(classes))
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
    Robust Grad-CAM:
    - registers forward hooks on ALL Conv2d layers,
    - picks the last layer that produced activations,
    - computes gradients and CAM, returns overlay (RGB numpy) and heatmap (0..1 float).
    """
    model.eval()
    x = preprocess_pil(pil_img).to(device)  # 1,C,H,W

    # find all conv modules
    convs = find_conv_modules(model)
    if not convs:
        raise RuntimeError("No Conv2d modules found in model for Grad-CAM.")

    activations_map = {}
    gradients_map = {}

    # forward hooks -> save activations
    forward_handles = []
    def make_fwd_hook(key):
        def hook(module, inp, out):
            # store a detached copy (we will need it on CPU later)
            activations_map[key] = out
        return hook

    # backward hooks -> save gradients
    backward_handles = []
    def make_bwd_hook(key):
        def hook(module, grad_in, grad_out):
            # grad_out[0] corresponds to gradient w.r.t. module output
            gradients_map[key] = grad_out[0]
        return hook

    # register hooks on all conv modules
    for name, mod in convs:
        forward_handles.append(mod.register_forward_hook(make_fwd_hook(name)))
        # use register_full_backward_hook if available, otherwise register_backward_hook
        try:
            backward_handles.append(mod.register_full_backward_hook(make_bwd_hook(name)))
        except AttributeError:
            backward_handles.append(mod.register_backward_hook(make_bwd_hook(name)))

    # forward pass
    logits = model(x)  # shape (1, num_classes)
    probs = torch.softmax(logits, dim=1)
    if target_class is None:
        target_class = int(probs.argmax(dim=1).item())

    # backward pass for the target class score
    model.zero_grad()
    # ensure scalar selection
    score = logits[:, target_class].squeeze()
    # if score is not scalar (rare), sum
    if score.dim() != 0:
        score = score.sum()
    score.backward(retain_graph=True)

    # remove hooks
    for h in forward_handles + backward_handles:
        try: h.remove()
        except Exception: pass

    # choose the last conv that actually produced activations and gradients
    last_key = None
    for name, _ in convs:
        if name in activations_map and name in gradients_map:
            last_key = name

    if last_key is None:
        # fallback: choose the last conv that produced activations even if gradients missing
        for name, _ in reversed(convs):
            if name in activations_map:
                last_key = name
                break

    if last_key is None:
        raise RuntimeError("Grad-CAM failed: no activations captured from any Conv2d layers.")

    act = activations_map[last_key]   # Tensor shape (1, C, H, W)
    grad = gradients_map.get(last_key, None)  # may be None in fallback

    if grad is None:
        # try to compute gradients by re-running backward on a small scalar loss
        raise RuntimeError(f"Gradients not captured for layer {last_key}; ensure model allows backprop to conv outputs.")

    # compute weights: global average pooling of gradients over spatial dims
    # shape: (1, C, 1, 1)
    weights = torch.mean(grad, dim=(2,3), keepdim=True)

    # weighted sum of activations
    gcam_map = torch.sum(weights * act, dim=1, keepdim=True)  # shape (1,1,H,W)
    gcam_map = torch.relu(gcam_map)
    gcam_map = gcam_map.squeeze().cpu().numpy()  # H x W

    # normalize to 0-1
    if gcam_map.max() > 0:
        gcam_map = (gcam_map - gcam_map.min()) / (gcam_map.max() - gcam_map.min() + 1e-8)
    else:
        gcam_map = np.zeros_like(gcam_map)

    # overlay on original image
    orig = np.array(pil_img.convert("RGB"))
    heatmap = cv2.resize(gcam_map, (orig.shape[1], orig.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    cmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
    overlay = (0.4 * cmap.astype(float) + 0.6 * orig.astype(float)).astype(np.uint8)

    return overlay, heatmap

# ---------- helper: tumor presence ----------
def tumor_present(pred_result):
    label = pred_result['pred_class'].lower()
    if label == 'notumor' or label == 'no tumor' or 'no' in label:
        return False
    return True
