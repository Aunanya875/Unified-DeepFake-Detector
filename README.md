# Unified-DeepFake-Detector
A unified deepfake detection framework for both image and video analysis. This repository includes a TensorFlow-based model leveraging the Xception architecture for shared feature extraction, combined with LSTM for temporal video processing.

### Key Features
- Unified architecture for both image and video deepfake detection
- Based on Xception backbone with additional LSTM layers for temporal analysis
- Supports both single-frame and multi-frame inputs
- Comprehensive logging and model checkpointing
- Modular design for easy maintenance and extension

## Project Structure
```
unified-deepfake-detection/
├── com/
│   └── mhire/
│       ├── data_processing/
│       │   └── data_processing.py
│       ├── training/
│       │   └── training.py
│       └── evaluation/
│           └── evaluation.py
├── main.py
├── requirements.txt
├── README.md
└── LICENSE
```

### Component Description
- **data_processing.py**: Handles data loading, preprocessing, and dataset splitting
- **training.py**: Contains the model architecture and training pipeline
- **evaluation.py**: Manages model evaluation and metrics calculation
- **main.py**: Entry point of the application, orchestrates the entire pipeline

## Technical Architecture
The system uses a dual-branch architecture:
1. **Image Branch**: Processes single frames using Xception backbone
2. **Video Branch**: Processes sequences using Xception + LSTM for temporal features
3. **Unified Output**: Combines both branches for comprehensive detection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/unified-deepfake-detection.git
cd unified-deepfake-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure
The system expects data in the following structure:
```
dataset/
├── real/
│   ├── video1/
│   │   ├── frame001.png
│   │   ├── frame002.png
│   │   └── ...
│   └── ...
└── fake/
    ├── video1/
    │   ├── frame001.png
    │   ├── frame002.png
    │   └── ...
    └── ...
```

## Usage

1. Update the paths in main.py:
```python
base_dir = Path("your/base/directory")
data_path = base_dir / "Datasets/FaceForensics"
processed_data_dir = base_dir / "ProcessedData/FaceForensics"
model_dir = base_dir / "Trained_models/Unified_DeepFake_Detection_Model"
```

2. Run the pipeline:
```bash
python main.py
```

The system will:
1. Process and split the dataset (70% train, 15% validation, 15% test)
2. Train the unified model
3. Evaluate performance and generate metrics

## GPU Support

The system automatically detects and utilizes available GPU resources. For optimal performance:
- CUDA-compatible GPU recommended
- Minimum 8GB GPU memory
- Updated GPU drivers

## Troubleshooting

Common issues and solutions:

1. **GPU Memory Error**:
   - Reduce batch size in training.py
   - Decrease image size or sequence length

2. **Data Loading Error**:
   - Verify dataset structure
   - Check file permissions
   - Ensure correct path configuration

3. **Training Instability**:
   - Adjust learning rate
   - Modify batch size
   - Check for data imbalance

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Citation
If you use this code in your research, please cite:
```
@project{unified_deepfake_detection,
  title = {Unified DeepFake Detection System},
  year = {2025},
  author = {Syeda Aunanya Mahmud},
  url = {https://github.com/Aunanya875/Unified-DeepFake-Detector}
}
```
