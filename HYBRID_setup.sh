#!/bin/bash

echo "========================================"
echo "Brain Tumor Detection - Hybrid Setup"
echo "========================================"
echo ""

echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment"
    exit 1
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Installing core dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Installing interface dependencies..."
pip install -r interface/requirements.txt

echo ""
echo "Creating necessary directories..."
mkdir -p checkpoints
mkdir -p data
mkdir -p interface/uploads

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To train the model, run:"
echo "  python src/train.py --train_dir data/Training --test_dir data/Testing --use_qnn"
echo ""
echo "To start the web interface, run:"
echo "  cd interface"
echo "  python app.py"
echo ""
