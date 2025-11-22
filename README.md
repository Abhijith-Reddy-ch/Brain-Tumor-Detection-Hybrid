# 🧠 Brain Tumor Detection with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.2.5-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A deep learning system for **detecting brain tumors from MRI images** using **EfficientNet-B0 CNN architecture**, achieving **95%+ accuracy**. Features a modern web interface with real-time predictions and Grad-CAM visualizations.

## 🎯 Key Features

- 🤖 **CNN Architecture**: EfficientNet-B0 with transfer learning
- 🎯 **95%+ Accuracy**: High performance on brain tumor classification
- 🌐 **Modern Web Interface**: Beautiful dark-themed UI with drag-and-drop upload
- 🔥 **Grad-CAM Visualization**: Explainable AI showing model attention regions
- 📊 **4-Class Classification**: Glioma, Meningioma, Pituitary, No Tumor

> ⚠️ **Disclaimer**: This project is for **educational and research purposes only**.  
> **NOT** for medical diagnosis or clinical use.

## 🧠 Tumor Classes

| Class | Description |
|-------|-------------|
| **Glioma** | Cancerous tumor in glial cells |
| **Meningioma** | Tumor from meninges |
| **Pituitary** | Pituitary gland tumor |
| **No Tumor** | Normal brain MRI |

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Abhijith-Reddy-ch/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

### 2. Install Dependencies

**Windows:**
```bash
scripts\setup.bat
```

**Linux/Mac:**
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

**Manual Installation:**
```bash
pip install -r requirements.txt
pip install -r interface/requirements.txt
```

### 3. Download Dataset
Download the Brain Tumor MRI Dataset from [Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

### 4. Train the Model (Optional)
```bash
python src/train.py \
  --train_dir "data/Training" \
  --test_dir "data/Testing" \
  --epochs 20
```

### 5. Download Pre-trained Weights
Download from [Releases](https://github.com/Abhijith-Reddy-ch/Brain-Tumor-Detection/releases/latest) and place in:
```
checkpoints/best.pth
```

### 6. Run the Web Interface
```bash
cd interface
python app.py
```

Open **http://localhost:5000** in your browser.

## 📂 Project Structure

```
Brain-Tumor-Detection/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Core dependencies
│
├── src/                         # Source code
│   ├── train.py                # Training script
│   ├── model.py                # CNN model
│   ├── dataset.py              # Data loading & augmentation
│   └── export_onnx.py          # ONNX export utility
│
├── interface/                   # Web application
│   ├── app.py                  # Flask server
│   ├── model_utils.py          # Model loading & Grad-CAM
│   ├── requirements.txt        # Interface dependencies
│   ├── templates/
│   │   └── index.html          # Modern UI
│   └── uploads/                # Upload directory (gitignored)
│
├── scripts/                     # Utility scripts
│   ├── setup.sh                # Linux/Mac setup
│   └── setup.bat               # Windows setup
│
└── checkpoints/                 # Model weights (gitignored)
    └── .gitkeep
```

## 🎓 Model Architecture

### EfficientNet-B0 CNN
- **Base Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Input**: 224×224 RGB MRI scans
- **Output**: 4-class softmax predictions
- **Fine-tuning**: All layers trainable

### Training Configuration
- **Optimizer**: AdamW (lr=2e-4, weight_decay=1e-5)
- **Loss**: CrossEntropyLoss with label smoothing (0.1)
- **Scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)
- **Mixed Precision**: AMP for faster training on CUDA
- **Data Augmentation**: Albumentations pipeline
- **Class Balancing**: WeightedRandomSampler

## 📊 Performance

Achieves **95%+ accuracy** on the test set with strong performance across all tumor classes.

## 🧪 Training Your Own Model

### Basic Training
```bash
python src/train.py \
  --train_dir "data/Training" \
  --test_dir "data/Testing" \
  --epochs 20 \
  --batch_size 32
```

### Training Arguments
- `--train_dir`: Path to training data directory
- `--test_dir`: Path to testing data directory
- `--ckpt_dir`: Directory to save checkpoints (default: ./checkpoints)
- `--img_size`: Input image size (default: 224)
- `--batch_size`: Batch size for training (default: 32)
- `--num_workers`: DataLoader workers (default: 0)
- `--lr`: Learning rate (default: 2e-4)
- `--epochs`: Number of training epochs (default: 20)

## 🌐 Web Interface Features

### Modern UI Design
- 🌙 **Dark Theme**: Gradient background with glassmorphism effects
- 📁 **Drag & Drop**: Intuitive file upload with preview
- 🎨 **Animations**: Smooth transitions and micro-interactions
- 📱 **Responsive**: Mobile-friendly design
- 🔥 **Grad-CAM**: Visual explanation of predictions
- 📊 **Probability Bars**: Gradient-filled visualization

### Usage
1. Navigate to `http://localhost:5000`
2. Upload an MRI scan (PNG, JPG, JPEG - max 10MB)
3. View instant prediction results
4. Analyze confidence scores and class probabilities
5. Examine Grad-CAM heatmap showing model attention

## 📦 Dataset

We use the **Brain Tumor MRI Dataset** from Kaggle:
- 🔗 [Kaggle Dataset Link](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- **Training**: 2,439 images
- **Testing**: 431 images
- **Classes**: 4 (Glioma, Meningioma, Pituitary, No Tumor)

## 🛠️ Requirements

### Core Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
albumentations>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
scikit-learn>=1.2.0
opencv-python>=4.7.0
```

### Web Interface
```
flask==2.2.5
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.5.0
numpy>=1.24.0
opencv-python>=4.7.0
```

## 🚀 Export to ONNX

```bash
python src/export_onnx.py \
  --ckpt "./checkpoints/best.pth" \
  --out "./checkpoints/brain_tumor_model.onnx" \
  --img_size 224
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔐 Ethical Use

This model is:
- ❌ **NOT** FDA-approved or medically certified
- ❌ **NOT** intended for clinical diagnosis
- ❌ **NOT** a replacement for professional medical advice
- ✅ **ONLY** for academic and research purposes

## 🌟 Future Enhancements

- [ ] TensorRT acceleration
- [ ] Mobile app (iOS/Android)
- [ ] React-based frontend
- [ ] Multi-model ensemble
- [ ] 3D MRI volume analysis
- [ ] Tumor segmentation with UNet
- [ ] Model quantization for edge devices

## 🙏 Acknowledgments

- **Dataset**: [Sartaj Bhuvaji et al.](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) - Kaggle Brain Tumor MRI Dataset
- **Base Model**: [EfficientNet](https://arxiv.org/abs/1905.11946) (Tan & Le, 2019)
- **Explainability**: [Grad-CAM](https://arxiv.org/abs/1610.02391) (Selvaraju et al., 2017)
- **Maintainer**: [Abhijith Reddy Ch](https://github.com/Abhijith-Reddy-ch)

## 📧 Contact

For questions or collaborations, please open an issue or reach out via GitHub.

---

⭐ If you find this project helpful, please consider giving it a star!

Made with ❤️ and 🤖 (Deep Learning)
