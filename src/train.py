import os
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from model import unet_model
from data_loader import OilSpillDataset

def train(image_dir, mask_dir, val_image_dir, val_mask_dir, epochs=20, batch_size=16, model_save_path="models/best_model.keras"):
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Data Loaders
    train_gen = OilSpillDataset(image_dir, mask_dir, batch_size=batch_size)
    val_gen = OilSpillDataset(val_image_dir, val_mask_dir, batch_size=batch_size, shuffle=False)

    # Model
    model = unet_model(input_size=(256, 256, 3))
    
    # Callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train
    print("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, early_stop]
    )

    # Plot History
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.savefig('models/training_history.png')
    print("Training finished. History saved to models/training_history.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Using absolute paths as defaults for convenience in this environment
    default_train_img = r"c:/Users/banda/oilspilldet/images-20251221T121915Z-3-001/images/images/train"
    default_train_mask = r"c:/Users/banda/oilspilldet/masks-20251221T121949Z-3-001/masks/masks/train"
    
    # Assuming there are validation folders, if not we might split train_gen internally or user must provide
    # Based on file listing, there IS a 'val' folder in the parent `images` and `masks` directories.
    # We should verify that structure again. 
    # Previous list_dir showed `train` and `val` inside `images-20251221T121915Z-3-001/images/images/`
    
    default_val_img = r"c:/Users/banda/oilspilldet/images-20251221T121915Z-3-001/images/images/val"
    default_val_mask = r"c:/Users/banda/oilspilldet/masks-20251221T121949Z-3-001/masks/masks/val"

    parser.add_argument("--image_dir", type=str, default=default_train_img)
    parser.add_argument("--mask_dir", type=str, default=default_train_mask)
    parser.add_argument("--val_image_dir", type=str, default=default_val_img)
    parser.add_argument("--val_mask_dir", type=str, default=default_val_mask)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8) # Lower batch size to be safe with memory

    args = parser.parse_args()
    
    train(args.image_dir, args.mask_dir, args.val_image_dir, args.val_mask_dir, epochs=args.epochs, batch_size=args.batch_size)
