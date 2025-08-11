"""
Training diagnostics to identify issues with low accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from config import Config
from processdata import DataProcessor
from trainmodel import ModelTrainer

class TrainingDiagnostics:
    def __init__(self):
        self.processor = DataProcessor()
        
    def analyze_dataset(self, data_path):
        """Analyze the dataset for common issues"""
        print("=== DATASET ANALYSIS ===")
        
        # Load or process data
        try:
            X, y, norm_params = self.processor.load_processed_data()
            if X is None:
                print("No processed data found. Processing dataset...")
                X, y, norm_params = self.processor.prepare_training_data(data_path, display_videos=False)
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None, None
        
        # Basic statistics
        print(f"Dataset shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Feature dimension: {X.shape[2]}")
        print(f"Sequence length: {X.shape[1]}")
        
        # Class distribution
        fight_samples = np.sum(y == 1)
        normal_samples = np.sum(y == 0)
        total_samples = len(y)
        
        print(f"\nClass Distribution:")
        print(f"Fight samples: {fight_samples} ({fight_samples/total_samples*100:.1f}%)")
        print(f"Normal samples: {normal_samples} ({normal_samples/total_samples*100:.1f}%)")
        
        # Check for class imbalance
        if abs(fight_samples - normal_samples) / total_samples > 0.3:
            print("⚠️  WARNING: Significant class imbalance detected!")
            print("   Consider using class weights or data augmentation")
        
        # Analyze feature distributions
        print(f"\nFeature Analysis:")
        
        # Check for NaN or infinite values
        nan_count = np.sum(np.isnan(X))
        inf_count = np.sum(np.isinf(X))
        
        print(f"NaN values: {nan_count}")
        print(f"Infinite values: {inf_count}")
        
        if nan_count > 0 or inf_count > 0:
            print("⚠️  WARNING: Invalid values found in features!")
        
        # Check feature variance
        feature_vars = np.var(X, axis=(0, 1))
        low_var_features = np.sum(feature_vars < 1e-6)
        
        print(f"Low variance features: {low_var_features}/{len(feature_vars)}")
        
        if low_var_features > 5:
            print("⚠️  WARNING: Many features have very low variance!")
        
        # Analyze sequence lengths (non-zero frames)
        sequence_lengths = []
        for i in range(X.shape[0]):
            # Count non-zero frames (assuming zero padding)
            non_zero_frames = np.sum(np.any(X[i] != 0, axis=1))
            sequence_lengths.append(non_zero_frames)
        
        sequence_lengths = np.array(sequence_lengths)
        
        print(f"\nSequence Length Analysis:")
        print(f"Mean sequence length: {np.mean(sequence_lengths):.1f}")
        print(f"Min sequence length: {np.min(sequence_lengths)}")
        print(f"Max sequence length: {np.max(sequence_lengths)}")
        
        # Check if sequences are too short
        short_sequences = np.sum(sequence_lengths < Config.MIN_SEQUENCE_LENGTH)
        print(f"Sequences shorter than {Config.MIN_SEQUENCE_LENGTH}: {short_sequences}")
        
        if short_sequences > total_samples * 0.2:
            print("⚠️  WARNING: Many sequences are very short!")
        
        # Feature-wise analysis
        print(f"\nFeature-wise Analysis:")
        
        # Split features into angles and velocities
        angles = X[:, :, :4]  # First 4 features are angles
        velocities = X[:, :, 4:]  # Rest are velocities
        
        # Analyze angles
        angle_means = np.mean(angles, axis=(0, 1))
        angle_stds = np.std(angles, axis=(0, 1))
        
        print(f"Angle features:")
        angle_names = ['Left arm', 'Right arm', 'Left leg', 'Right leg']
        for i, name in enumerate(angle_names):
            print(f"  {name}: mean={angle_means[i]:.1f}°, std={angle_stds[i]:.1f}°")
        
        # Analyze velocities
        velocity_means = np.mean(velocities, axis=(0, 1))
        velocity_stds = np.std(velocities, axis=(0, 1))
        
        print(f"Velocity features:")
        print(f"  Mean velocity: {np.mean(velocity_means):.3f}")
        print(f"  Std velocity: {np.mean(velocity_stds):.3f}")
        
        # Check for potential issues
        recommendations = []
        
        if fight_samples < 50:
            recommendations.append("Collect more fight samples")
        
        if normal_samples < 50:
            recommendations.append("Collect more normal samples")
        
        if abs(fight_samples - normal_samples) / total_samples > 0.3:
            recommendations.append("Balance the dataset or use class weights")
        
        if np.mean(sequence_lengths) < 20:
            recommendations.append("Increase sequence length or collect longer videos")
        
        if low_var_features > 5:
            recommendations.append("Check feature extraction - many features have low variance")
        
        if np.mean(velocity_means) < 0.001:
            recommendations.append("Velocity features seem too small - check normalization")
        
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        return X, y, norm_params
    
    def improved_model_config(self):
        """Return improved model configuration"""
        print("\n=== IMPROVED MODEL CONFIGURATION ===")
        
        improved_config = {
            'learning_rate': 0.0005,  # Lower learning rate
            'batch_size': 16,         # Larger batch size
            'epochs': 50,             # More epochs
            'early_stopping_patience': 10,  # More patience
            'dropout_rate': 0.3,      # Adjusted dropout
            'lstm_units': [128, 64],  # Larger LSTM units
            'dense_units': 32,        # Larger dense layer
        }
        
        print("Recommended hyperparameters:")
        for key, value in improved_config.items():
            print(f"  {key}: {value}")
        
        return improved_config
    
    def train_improved_model(self, X, y):
        """Train model with improved configuration"""
        print("\n=== TRAINING IMPROVED MODEL ===")
        
        from sklearn.model_selection import train_test_split
        from sklearn.utils.class_weight import compute_class_weight
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Masking
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        from tensorflow.keras.optimizers import Adam
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Calculate class weights to handle imbalance
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        print(f"Class weights: {class_weight_dict}")
        
        # Build improved model
        model = Sequential([
            Masking(mask_value=0.0, input_shape=(X.shape[1], X.shape[2])),
            
            # First LSTM layer
            LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            BatchNormalization(),
            
            # Second LSTM layer
            LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            BatchNormalization(),
            
            # Third LSTM layer
            LSTM(32, dropout=0.3, recurrent_dropout=0.3),
            BatchNormalization(),
            
            # Dense layers
            Dense(32, activation='relu'),
            Dropout(0.4),
            Dense(16, activation='relu'),
            Dropout(0.3),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile with improved settings
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("Model architecture:")
        model.summary()
        
        # Improved callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath='improved_fight_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=16,
            epochs=50,
            validation_data=(X_val, y_val),
            class_weight=class_weight_dict,  # Use class weights
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        val_loss, val_accuracy, val_precision, val_recall = model.evaluate(X_val, y_val, verbose=0)
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Precision: {val_precision:.4f}")
        print(f"Validation Recall: {val_recall:.4f}")
        
        if val_precision > 0 and val_recall > 0:
            f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall)
            print(f"Validation F1-Score: {f1_score:.4f}")
        
        # Detailed predictions
        y_pred = model.predict(X_val, verbose=0)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        print(f"\nClassification Report:")
        print(classification_report(y_val, y_pred_binary, target_names=['Normal', 'Fight']))
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_val, y_pred_binary)
        print(cm)
        
        # Plot training history
        self.plot_training_history(history)
        
        return model, history
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training')
        axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Training')
        axes[1, 0].plot(history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Training')
        axes[1, 1].plot(history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    diagnostics = TrainingDiagnostics()
    
    # Analyze dataset
    X, y, norm_params = diagnostics.analyze_dataset(Config.DATA_PATH)
    
    if X is not None and y is not None:
        # Get improved configuration
        improved_config = diagnostics.improved_model_config()
        
        # Ask user if they want to train improved model
        response = input("\nWould you like to train the improved model? (y/n): ")
        if response.lower() == 'y':
            model, history = diagnostics.train_improved_model(X, y)
            print("\nImproved model saved as 'improved_fight_model.keras'")
            print("Update your Config.MODEL_PATH to use this model!")
    else:
        print("Could not analyze dataset. Please check your data path and files.")