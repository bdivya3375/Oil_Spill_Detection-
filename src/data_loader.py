import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class OilSpillDataset(tf.keras.utils.Sequence):
    def __init__(self, image_dir, mask_dir, batch_size=16, img_size=(256, 256), shuffle=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])
        self.mask_filenames = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])
        
        # Simple check to ensure alignment
        if len(self.image_filenames) != len(self.mask_filenames):
            print(f"Warning: Number of images ({len(self.image_filenames)}) and masks ({len(self.mask_filenames)}) do not match!")
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_names = [self.image_filenames[k] for k in indexes]
        batch_mask_names = [self.mask_filenames[k] for k in indexes]
        
        X, y = self.__data_generation(batch_image_names, batch_mask_names)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_image_names, batch_mask_names):
        X = np.empty((self.batch_size, *self.img_size, 3))
        y = np.empty((self.batch_size, *self.img_size, 1))

        for i, (img_name, mask_name) in enumerate(zip(batch_image_names, batch_mask_names)):
            # Load Image
            img_path = os.path.join(self.image_dir, img_name)
            img = load_img(img_path, target_size=self.img_size)
            img = img_to_array(img) / 255.0  # Normalize
            X[i] = img

            # Load Mask
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = load_img(mask_path, target_size=self.img_size, color_mode="grayscale")
            mask = img_to_array(mask)
            mask = mask / 255.0
            mask[mask > 0.5] = 1.0
            mask[mask <= 0.5] = 0.0
            y[i] = mask

        return X, y
