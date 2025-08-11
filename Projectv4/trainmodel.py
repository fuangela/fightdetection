"""
LSTM model training module
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from processdata import DataProcessor
from config import Config

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.history = None
        
    def build_model(self, input_shape):
        """
        Build LSTM model for fight detection
        
        Args:
            input_shape: Tuple of (sequence_length, feature_dimension)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # Masking layer to handle padded sequences
            Masking(mask_value=0.0, input_shape=input_shape),
            
            # First LSTM layer with return sequences
            LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            
            # Second LSTM layer
            LSTM(32, dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            
            # Dense layers
            Dense(16, activation='relu'),
            Dropout(0.3),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_callbacks(self):
        """
        Create training callbacks
        
        Returns:
            List of callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=Config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=Config.MODEL_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, X, y, validation_split=None, test_size=None):
        """
        Train the LSTM model
        
        Args:
            X: Training data
            y: Training labels
            validation_split: Fraction for validation split
            test_size: Test set size for train_test_split
            
        Returns:
            Training history
        """
        if test_size is None:
            test_size = Config.TRAIN_TEST_SPLIT
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42, 
            stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Training fight samples: {np.sum(y_train == 1)}")
        print(f"Training no-fight samples: {np.sum(y_train == 0)}")
        
        # Build model
        input_shape = (X.shape[1], X.shape[2])
        self.model = self.build_model(input_shape)
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train model
        print("\nStarting training...")
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=Config.BATCH_SIZE,
            epochs=Config.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on validation set
        print("\nEvaluating model...")
        val_loss, val_accuracy, val_precision, val_recall = self.model.evaluate(
            X_val, y_val, verbose=0
        )
        
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Precision: {val_precision:.4f}")
        print(f"Validation Recall: {val_recall:.4f}")
        print(f"Validation F1-Score: {2 * (val_precision * val_recall) / (val_precision + val_recall):.4f}")
        
        # Generate predictions for detailed evaluation
        y_pred = self.model.predict(X_val, verbose=0)
        y_pred_binary = (y_pred > Config.FIGHT_CONFIDENCE_THRESHOLD).astype(int)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred_binary, target_names=['No Fight', 'Fight']))
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_val, y_pred_binary)
        print(cm)
        
        return self.history
    
    def plot_training_history(self, save_plot=True):
        """
        Plot training history
        
        Args:
            save_plot: Whether to save the plot
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return