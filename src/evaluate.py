import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from data_loader import OilSpillDataset

def get_metrics(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    
    return iou, dice

def evaluate(image_dir, mask_dir, model_path="models/best_model.keras", output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Model
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Data Loader
    # Using batch size 1 for detailed evaluation/visualization if needed, 
    # but for metrics aggregate we can use larger batch
    batch_size = 16 
    val_gen = OilSpillDataset(image_dir, mask_dir, batch_size=batch_size, shuffle=False)
    
    print("Evaluating...")
    
    total_iou = 0
    total_dice = 0
    count = 0
    
    # Iterate through all batches
    for i in range(len(val_gen)):
        X, y_true = val_gen[i]
        y_pred = model.predict(X, verbose=0)
        
        # Binarize predictions
        y_pred_binary = (y_pred > 0.5).astype(np.float32)
        
        # Metrics for this batch
        iou, dice = get_metrics(y_true, y_pred_binary)
        
        total_iou += iou
        total_dice += dice
        count += 1
        
        # Visualize first batch or first few images
        if i == 0:
            for j in range(min(5, len(X))):
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                plt.title("Satellite Image")
                plt.imshow(X[j])
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.title("Ground Truth Mask")
                plt.imshow(y_true[j].squeeze(), cmap='gray')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.title("Predicted Mask")
                plt.imshow(y_pred_binary[j].squeeze(), cmap='gray')
                plt.axis('off')
                
                plt.savefig(f"{output_dir}/result_{j}.png")
                plt.close()

    avg_iou = total_iou / count if count > 0 else 0
    avg_dice = total_dice / count if count > 0 else 0
    
    print(f"Evaluation Results:")
    print(f"Files Processed: {count * batch_size}") # approx
    print(f"Mean IoU: {avg_iou:.4f}")
    print(f"Mean Dice Coefficient: {avg_dice:.4f}")
    
    # Save metrics to file
    with open(f"{output_dir}/metrics.txt", "w") as f:
        f.write(f"Mean IoU: {avg_iou:.4f}\n")
        f.write(f"Mean Dice Coefficient: {avg_dice:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Defaulting to val directory based on assumed structure
    default_val_img = r"c:/Users/banda/oilspilldet/images-20251221T121915Z-3-001/images/images/val"
    default_val_mask = r"c:/Users/banda/oilspilldet/masks-20251221T121949Z-3-001/masks/masks/val"
    
    parser.add_argument("--image_dir", type=str, default=default_val_img)
    parser.add_argument("--mask_dir", type=str, default=default_val_mask)
    parser.add_argument("--model_path", type=str, default="models/best_model.keras")
    
    args = parser.parse_args()
    
    evaluate(args.image_dir, args.mask_dir, args.model_path)
