"""
Script to train the LSTM model
"""

from trainmodel import ModelTrainer
from processdata import DataProcessor
from config import Config

if __name__ == "__main__":
    print("Starting model training...")
    
    # Load processed data
    processor = DataProcessor()
    X, y, normalization_params = processor.load_processed_data()
    
    if X is None:
        print("No processed data found. Please run data processing first.")
        exit(1)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train model
    try:
        history = trainer.train_model(X, y)
        
        # Plot training history
        trainer.plot_training_history(save_plot=True)
        
        print("Model training completed successfully!")
        print(f"Model saved to: {Config.MODEL_PATH}")
        
    except Exception as e:
        print(f"Error during training: {e}")