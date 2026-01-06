import argparse
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np

def predict(image_path, output_path="prediction.png", model_path="models/best_model.keras"):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    print(f"Processing image {image_path}...")
    img = load_img(image_path, target_size=(256, 256))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    mask = (pred[0] > 0.5).astype(np.uint8) * 255
    
    mask_img = array_to_img(mask, scale=False)
    mask_img.save(output_path)
    print(f"Prediction saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--output_path", type=str, default="prediction.png", help="Path to save the predicted mask")
    parser.add_argument("--model_path", type=str, default="models/best_model.keras", help="Path to the trained model")
    
    args = parser.parse_args()
    
    predict(args.image_path, output_path=args.output_path, model_path=args.model_path)
