import logging
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

class Trainer:
    def __init__(self, model_dir, img_size=224, max_seq_length=20):
        self.model_dir = Path(model_dir)
        self.img_size = img_size
        self.max_seq_length = max_seq_length
        self.model = None
        
        # Create model directory
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup GPU
        self._setup_gpu()

    def _setup_gpu(self):
        """Setup GPU configuration."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info("Memory growth enabled for GPUs")
            except RuntimeError as e:
                logging.error(f"Error setting GPU memory growth: {e}")

    def build_model(self):
        """Builds the unified model."""
        logging.info("Building model")
        
        # Shared backbone
        backbone = Xception(weights='imagenet', include_top=False, pooling='avg', 
                          input_shape=(self.img_size, self.img_size, 3))
        
        # Image branch
        image_input = backbone.input
        x = Dense(512, activation='relu')(backbone.output)
        x = Dropout(0.5)(x)
        single_frame_prediction = Dense(1, activation='sigmoid', name="image_output")(x)

        # Video branch
        video_input = tf.keras.Input(shape=(self.max_seq_length, self.img_size, self.img_size, 3))
        frame_features = TimeDistributed(backbone)(video_input)
        frame_features = LSTM(256, return_sequences=False)(frame_features)
        frame_features = Dropout(0.5)(frame_features)
        video_prediction = Dense(1, activation='sigmoid', name="video_output")(frame_features)

        self.model = Model(inputs=[image_input, video_input], 
                         outputs=[single_frame_prediction, video_prediction])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss={'image_output': 'binary_crossentropy', 
                  'video_output': 'binary_crossentropy'},
            metrics={'image_output': 'accuracy', 
                    'video_output': 'accuracy'}
        )
        
        logging.info("Model built successfully")

    def train(self, train_data, val_data):
        """Train the model."""
        logging.info("Starting model training")
        
        callbacks = [
            ModelCheckpoint(
                filepath=self.model_dir / 'best_model.keras',
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            CSVLogger(str(self.model_dir / 'training_metrics.csv'))
        ]

        history = self.model.fit(
            x=[train_data['image_data'], train_data['video_data']],
            y=[train_data['labels'], train_data['labels']],
            validation_data=(
                [val_data['image_data'], val_data['video_data']],
                [val_data['labels'], val_data['labels']]
            ),
            batch_size=16,
            epochs=20,
            callbacks=callbacks
        )
        
        # Save final model
        self.model.save(self.model_dir / 'final_model.keras')
        logging.info("Training completed")
        return history