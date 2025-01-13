import logging
from pathlib import Path
from com.mhire.data_processing.data_processing import DataProcessor
from com.mhire.training.training import Trainer
from com.mhire.evaluation.evaluation import Evaluator

def setup_logging(log_dir):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=log_dir / 'pipeline.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    # Setup paths
    base_dir = Path("/tmp")
    data_path = base_dir / "Datasets/FaceForensics"
    processed_data_dir = base_dir / "ProcessedData/FaceForensics"
    model_dir = base_dir / "Trained_models/Unified_DeepFake_Detection_Model"
    log_dir = model_dir / "logs"

    # Setup logging
    setup_logging(log_dir)
    logging.info("Starting deepfake detection pipeline")

    try:
        # Step 1: Process Data
        processor = DataProcessor(data_path, processed_data_dir)
        processed_data = processor.process_and_save_data()
        logging.info("Data processing completed")

        # Step 2: Train Model
        trainer = Trainer(model_dir=model_dir)
        trainer.build_model()
        trainer.train(
            train_data=processed_data['train'],
            val_data=processed_data['val']
        )
        logging.info("Model training completed")

        # Step 3: Evaluate Model
        evaluator = Evaluator(model_path=model_dir / 'best_model.keras')
        evaluator.load_model()
        metrics = evaluator.evaluate(processed_data['test'])
        logging.info("Model evaluation completed")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()