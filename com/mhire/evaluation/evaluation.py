import os
import logging
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score

class Evaluator:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

    def load_model(self):
        """Load the trained model."""
        logging.info("Loading model from %s", self.model_path)
        self.model = load_model(self.model_path)

    def evaluate(self, test_data):
        """Evaluate the model on test data."""
        logging.info("Starting model evaluation")
        
        predictions = self.model.predict(
            [test_data['image_data'], test_data['video_data']], 
            batch_size=16
        )
        
        # Process predictions
        image_predictions = (predictions[0] > 0.5).astype(int).flatten()
        video_predictions = (predictions[1] > 0.5).astype(int).flatten()
        ground_truth = test_data['labels'].flatten()

        # Calculate and log metrics
        logging.info("Computing metrics")
        
        # Image metrics
        image_accuracy = accuracy_score(ground_truth, image_predictions)
        image_report = classification_report(ground_truth, image_predictions)
        logging.info("Image Classification Results:")
        logging.info(f"Accuracy: {image_accuracy}")
        logging.info("\n" + image_report)

        # Video metrics
        video_accuracy = accuracy_score(ground_truth, video_predictions)
        video_report = classification_report(ground_truth, video_predictions)
        logging.info("Video Classification Results:")
        logging.info(f"Accuracy: {video_accuracy}")
        logging.info("\n" + video_report)

        return {
            'image_accuracy': image_accuracy,
            'video_accuracy': video_accuracy,
            'image_report': image_report,
            'video_report': video_report
        }