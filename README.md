# Oil Spill Detection System

This project implements a U-Net based deep learning model to detect oil spills in satellite imagery.

## Project Structure

- `src/`: Source code for the project.
    - `data_loader.py`: Data loading and preprocessing.
    - `model.py`: U-Net model architecture.
    - `train.py`: Training script.
    - `evaluate.py`: Evaluation and visualization script.
- `app.py`: Streamlit web application.
- `models/`: Directory where trained models are saved.
- `results/`: Directory where evaluation results and visualizations are saved.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model, run:
```bash
python src/train.py --epochs 20 --batch_size 16
```
This will save the best model to `models/best_model.keras` and training history plot to `models/training_history.png`.

### Evaluation

To evaluate the model and generate visualizations, run:
```bash
python src/evaluate.py
```
Results (metrics and images) will be saved in `results/`.

### Web Application

To run the interactive web interface:
```bash
streamlit run app.py
```
Upload an image to see the oil spill detection in real-time.

## Dataset

The dataset is expected to be in `c:/Users/banda/oilspilldet/` with `images` and `masks` directories.
