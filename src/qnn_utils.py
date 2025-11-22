import pennylane as qml
import torch
import torch.nn as nn

def create_qnn_layer(n_qubits, n_layers=1):
    """
    Creates a PennyLane QNN layer.
    
    Args:
        n_qubits (int): Number of qubits (and input/output features).
        n_layers (int): Number of variational layers.
        
    Returns:
        qml.qnn.TorchLayer: A PyTorch-compatible quantum layer.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        # Encoding inputs into quantum state
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        
        # Variational layers
        qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
        
        # Measurement
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    # Weight shape: (n_layers, n_qubits)
    weight_shapes = {"weights": (n_layers, n_qubits)}
    
    return qml.qnn.TorchLayer(circuit, weight_shapes)

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers=2):
        super(QuantumLayer, self).__init__()
        self.qnn = create_qnn_layer(n_qubits, n_layers)
        
    def forward(self, x):
        print(".", end="", flush=True) # Simple progress indicator
        return self.qnn(x)