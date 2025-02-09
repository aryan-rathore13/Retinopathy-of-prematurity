# Retinopathy of Prematurity Classification

## Overview
This project implements a deep learning model for automated classification of Retinopathy of Prematurity (ROP) using retinal images. The system leverages the VGG19 architecture and TensorFlow framework to provide accurate detection and classification of ROP cases.

### Key Features
- Utilizes VGG19 architecture for feature extraction
- Implements data augmentation for improved model generalization
- Achieves 96.3% classification accuracy
- Includes early stopping mechanism to prevent overfitting
- Uses ReLU activation and Adam optimizer for efficient training

## Technical Details
- **Framework:** TensorFlow
- **Programming Language:** Python
- **Base Architecture:** VGG19
- **Model Performance:**
  - Accuracy: 96.3%
  - Loss: 0.055

## Prerequisites
- Python 3.8+
- TensorFlow 2.x
- CUDA (for GPU support)
- Jupyter Notebook

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/rop-classification.git
cd rop-classification
```

2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

## Usage

1. Launch Jupyter Notebook
```bash
jupyter notebook
```

2. Open the `ROP_Classification.ipynb` notebook

3. Follow the instructions in the notebook to:
   - Load and preprocess the retinal images
   - Train the model
   - Evaluate the results
   - Make predictions on new images

## Model Architecture
The project uses a transfer learning approach with VGG19 as the base model:
- Pre-trained VGG19 for feature extraction
- Custom classification layers
- ReLU activation functions
- Adam optimizer for training optimization

## Results
The model achieves:
- 96.3% classification accuracy
- 0.055 loss value
- Detailed performance metrics available in the confusion matrix and classification report

## Contributing
Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT License](LICENSE)

## Contact
Your Name - [Your Email]
Project Link: [https://github.com/yourusername/rop-classification](https://github.com/yourusername/rop-classification)

## Acknowledgments
- VGG19 architecture
- TensorFlow team
- Contributors to the project
