"""
Script to process training data
"""


from processdata import DataProcessor
from config import Config

if __name__ == "__main__":
    print("Starting data processing with visual annotations...")
    
    # Initialize processor
    processor = DataProcessor()
    
    # Process dataset and prepare training data with visual display
    try:
        X, y, normalization_params = processor.prepare_training_data(
            Config.DATA_PATH, 
            save_data=True,
            display_videos=True  # Enable visual annotations
        )
        
        print("Data processing completed successfully!")
        print(f"Processed {len(X)} sequences")
        print(f"Feature dimension: {X.shape[2]}")
        print(f"Max sequence length: {X.shape[1]}")
        
    except Exception as e:
        print(f"Error during data processing: {e}")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")

'''
from processdata import DataProcessor
from config import Config

if __name__ == "__main__":
    print("Starting data processing...")
    
    # Initialize processor
    processor = DataProcessor()
    
    # Process dataset and prepare training data
    try:
        X, y, normalization_params = processor.prepare_training_data(
            Config.DATA_PATH, 
            save_data=True
        )
        
        print("Data processing completed successfully!")
        print(f"Processed {len(X)} sequences")
        print(f"Feature dimension: {X.shape[2]}")
        print(f"Max sequence length: {X.shape[1]}")
        
    except Exception as e:
        print(f"Error during data processing: {e}")
'''