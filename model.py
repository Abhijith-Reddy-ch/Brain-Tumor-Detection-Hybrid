# model.py
print("Starting model.py...")
import torch
print("Imported torch in model")
import torch.nn as nn
print("Imported torch.nn in model")

from qnn_utils import QuantumLayer
print("Imported QuantumLayer in model")

def create_model(num_classes, dropout=0.5, use_timm=False, use_qnn=False):
    """
    Returns model. Tries torchvision EfficientNet-B0, falls back to timm.
    """
    try:
        from torchvision import models
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        in_features = model.classifier[1].in_features
        if use_qnn:
            # Residual Hybrid Classifier
            # Path A: Classical (Linear -> Classes)
            # Path B: Quantum (Linear -> ReLU -> Linear -> QNN -> Linear)
            
            class ResidualHybridClassifier(nn.Module):
                def __init__(self, in_features, num_classes, n_qubits=8):
                    super().__init__()
                    # Classical path - Simplified to reduce overfitting
                    self.classical = nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(in_features, num_classes)
                    )
                    # Quantum path
                    self.quantum_pre = nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(in_features, 512),
                        nn.ReLU(),
                        nn.Linear(512, n_qubits)
                    )
                    self.qnn = QuantumLayer(n_qubits=n_qubits, n_layers=2)
                    self.quantum_post = nn.Linear(n_qubits, num_classes)
                    
                def forward(self, x):
                    # Classical output
                    out_c = self.classical(x)
                    
                    # Quantum output
                    x_q = self.quantum_pre(x)
                    x_q = self.qnn(x_q)
                    out_q = self.quantum_post(x_q)
                    
                    # Combine
                    return out_c + out_q

            model.classifier = ResidualHybridClassifier(in_features, num_classes)
        else:
            model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    except Exception:
        # fallback to timm
        import timm
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
    return model
