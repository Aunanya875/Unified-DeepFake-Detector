import os
import cv2
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, data_path, processed_data_dir, img_size=224, max_seq_length=20):
        self.data_path = Path(data_path)
        self.processed_data_dir = Path(processed_data_dir)
        self.img_size = img_size
        self.max_seq_length = max_seq_length
        
        # Create processed data directory
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.info("Initializing DataProcessor")

    def load_dataset(self):
        """Load dataset from the given data path."""
        logging.info("Loading dataset from %s", self.data_path)
        data = []
        for folder in self.data_path.iterdir():
            if folder.is_dir():
                label = 1 if 'FAKE' in folder.name.upper() else 0
                data.append({'folder': str(folder), 'label': label})
        return pd.DataFrame(data)

    def preprocess_images(self, folder):
        """Preprocess all images in a folder."""
        image_files = list(Path(folder).glob('*.png'))
        images = []

        for file in image_files[:self.max_seq_length]:
            img = cv2.imread(str(file))
            if img is not None:
                img = cv2.resize(img, (self.img_size, self.img_size))
                images.append(img / 255.0)

        while len(images) < self.max_seq_length:
            images.append(np.zeros((self.img_size, self.img_size, 3)))

        return np.array(images)

    def process_and_save_data(self):
        """Process and save the entire dataset."""
        logging.info("Starting data processing")
        
        # Load and split dataset
        dataset = self.load_dataset()
        train_df, temp_df = train_test_split(dataset, test_size=0.3, stratify=dataset['label'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

        # Process each split
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

        processed_data = {}
        for split_name, df in splits.items():
            logging.info(f"Processing {split_name} split")
            image_data, video_data, labels = [], [], []
            
            for _, row in df.iterrows():
                frames = self.preprocess_images(row['folder'])
                single_image = frames[0:1]
                image_data.append(single_image)
                video_data.append(frames)
                labels.append(row['label'])

            processed_split = {
                'image_data': np.array(image_data),
                'video_data': np.array(video_data),
                'labels': np.array(labels)
            }
            
            # Save processed split
            save_path = self.processed_data_dir / f'{split_name}_data.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(processed_split, f)
            
            processed_data[split_name] = processed_split
            logging.info(f"Saved {split_name} split to {save_path}")

        return processed_data
