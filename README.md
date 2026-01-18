<div align="center">

# Oil Spill Detection System

**Deep Learning-Powered Environmental Monitoring for Marine Oil Spill Detection**

[![Status](https://img.shields.io/badge/status-active%20development-success)](https://github.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)](https://streamlit.io/)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Use Cases](#use-cases)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **Oil Spill Detection System** is an advanced deep learning application designed to automatically identify and segment oil spills in satellite imagery. By leveraging state-of-the-art computer vision techniques, this project provides a reliable tool for environmental monitoring and disaster response, helping to mitigate the ecological impact of marine oil spills.

### Key Highlights

| **Aspect** | **Details** |
|:-----------|:------------|
| **Model Architecture** | U-Net Convolutional Neural Network |
| **Application Domain** | Computer Vision / Environmental Science |
| **Primary Use Case** | Automated Oil Spill Detection from Satellite Imagery |
| **Output Format** | Binary Segmentation Masks |
| **Interface** | Interactive Streamlit Web Application |

---

## Features

| **Feature** | **Description** |
|:------------|:----------------|
| **Accurate Segmentation** | Precise pixel-level detection of oil spills using U-Net architecture |
| **Interactive Interface** | User-friendly dashboard built with Streamlit for easy image analysis |
| **Real-time Processing** | Instant generation of prediction masks and overlay comparisons |
| **Spill Analysis** | Automatic calculation of spill percentage relative to image area |
| **Smart Alerts** | Visual alerts triggered when significant spills (> threshold) are detected |
| **Visualization Tools** | Comprehensive visualization with original images, masks, and overlays |

---

## Technologies

### Core Technologies

| **Category** | **Technologies** |
|:-------------|:----------------|
| **Programming Language** | Python 3.8+ |
| **Deep Learning Framework** | TensorFlow 2.x, Keras |
| **Web Framework** | Streamlit |
| **Data Processing** | NumPy, Pillow (PIL), OpenCV |
| **Visualization** | Matplotlib, Seaborn |

### Dependencies

```txt
tensorflow>=2.10.0
streamlit>=1.25.0
numpy>=1.23.0
pillow>=9.0.0
matplotlib>=3.6.0
seaborn>=0.12.0
openc-python>=4.6.0
```

---

## Architecture

### U-Net Model Overview

The system utilizes a **U-Net** architecture, a deep learning model renowned for its effectiveness in image segmentation tasks, adapted here for environmental monitoring.

#### Model Components

| **Component** | **Function** |
|:--------------|:-------------|
| **Encoder** | Captures context and features from input satellite images |
| **Decoder** | Enables precise localization of identified features |
| **Skip Connections** | Preserves spatial information lost during downsampling |
| **Output Layer** | Generates binary mask (oil spill vs. background) |

#### Architecture Flow

```
Input Image → Encoder (Feature Extraction) → Decoder (Upsampling) → Binary Mask Output
                ↓                              ↑
            Skip Connections (Feature Preservation)
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/oilspilldet.git
   cd oilspilldet
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

---

## Usage

### Training the Model

To train the model on your dataset:

```bash
python src/train.py --epochs 20 --batch_size 16 --learning_rate 0.001
```

**Training Parameters:**

| **Parameter** | **Default** | **Description** |
|:--------------|:------------|:----------------|
| `--epochs` | 20 | Number of training epochs |
| `--batch_size` | 16 | Batch size for training |
| `--learning_rate` | 0.001 | Learning rate for optimizer |
| `--data_path` | `images/` | Path to training images |

### Running the Web Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - Upload a satellite image through the interface
   - View the prediction results in real-time

### Command-Line Prediction

For batch processing:

```bash
python src/predict.py --input_dir images/ --output_dir results/
```

---

## Project Structure

```
oilspilldet/
├── app.py                 # Main Streamlit application entry point
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
│
├── src/                   # Source code modules
│   ├── data_loader.py     # Data preprocessing and loading utilities
│   ├── model.py           # U-Net architecture definition
│   ├── train.py           # Model training script
│   └── predict.py         # Prediction and inference logic
│
├── models/                # Saved model checkpoints
│   └── best_model.keras   # Trained model weights
│
├── results/               # Evaluation results and outputs
│   ├── predictions/       # Generated prediction masks
│   └── visualizations/    # Visualization outputs
│
├── images/                # Training and test images
│   ├── train/             # Training dataset
│   ├── test/              # Test dataset
│   └── masks/             # Ground truth masks
│
└── docs/                  # Additional documentation
    └── architecture.md    # Detailed architecture documentation
```

### Key Files Description

| **File / Directory** | **Purpose** |
|:---------------------|:------------|
| `app.py` | Main entry point for the Streamlit web application with UI components |
| `src/model.py` | Defines the U-Net neural network architecture and model compilation |
| `src/train.py` | Script to train the model on the dataset with configurable hyperparameters |
| `src/predict.py` | Contains logic for loading trained models and making predictions |
| `src/data_loader.py` | Handles data preprocessing, augmentation, and batch loading |
| `models/` | Stores trained model files (`.keras` or `.h5` format) |
| `results/` | Contains prediction outputs, evaluation metrics, and visualizations |

---

## Results

### Model Performance

The model generates binary segmentation masks with high accuracy for oil spill detection. The interface displays:

- **Original Image**: The uploaded satellite image
- **Predicted Mask**: The segmentation output highlighting oil spill regions
- **Overlay Visualization**: A blended view for easy verification
- **Spill Statistics**: Percentage of image area covered by detected spills

### Visualization Example

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Original Image │  │  Predicted Mask │  │  Overlay View   │
│                 │  │                 │  │                 │
│   [Satellite]   │  │   [Segmentation]│  │   [Blended]     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Demo


**[Watch Demo Video](images/oilspill.mp4)**

*(Note: To embed a playable video player directly on GitHub, please upload the video to a GitHub Issue/PR comment and copy the generated asset link here.)*


**Live Demo:** [Deploy to Streamlit Cloud](https://share.streamlit.io/)

---

## Use Cases

| **Scenario** | **Application** | **Impact** |
|:-------------|:----------------|:-----------|
| **Environmental Protection** | Early warning system to protect marine ecosystems | Prevents ecological damage through rapid detection |
| **Maritime Surveillance** | Monitoring shipping lanes for illegal dumping and accidental leaks | Enables proactive enforcement and monitoring |
| **Disaster Response** | Assessing spill magnitude to coordinate cleanup efforts | Optimizes resource allocation for emergency response |
| **Satellite Monitoring** | Automated analysis of large-scale satellite imagery | Scales monitoring capabilities efficiently |

---

## Future Enhancements

| **Enhancement** | **Impact** | **Priority** |
|:----------------|:-----------|:-------------|
| **Live Satellite Feeds** | Enable real-time monitoring of specific aquatic regions | High |
| **Multi-class Segmentation** | Distinguish between oil, algae, and other pollutants | Medium |
| **Mobile App Support** | Allow field reporters to upload images directly from mobile devices | Medium |
| **Cloud Deployment** | Scale the backend to process massive datasets efficiently | High |
| **API Integration** | Provide RESTful API for third-party integrations | Low |
| **Advanced Analytics** | Historical trend analysis and spill pattern recognition | Medium |

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Authors

| **Name** | **Role** | **Contact** |
|:---------|:---------|:------------|
| **[Your Name]** | Lead Developer | [Email](mailto:your.email@example.com) \| [GitHub](https://github.com/yourusername) |
| **[Teammate Name]** | Researcher | [Email](mailto:teammate@example.com) \| [GitHub](https://github.com/teammate) |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- U-Net architecture by Ronneberger et al. (2015)
- TensorFlow and Keras communities
- Streamlit for the web framework
- All contributors and researchers in the field of environmental monitoring

---

<div align="center">

**If you find this project useful, please consider giving it a star!**

Made with for environmental protection

</div>
