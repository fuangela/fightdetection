"""
Feature extraction utilities for pose-based fight detection
"""

import numpy as np
from config import Config

class FeatureExtractor:
    def __init__(self):
        self.angle_keypoints = Config.ANGLE_KEYPOINTS
    
    def calculate_angle(self, point_a, point_b, point_c):
        """
        Calculate angle between three points
        
        Args:
            point_a, point_b, point_c: Points as [y, x, confidence] arrays
            
        Returns:
            Angle in degrees (0-180)
        """
        # Convert to numpy arrays and extract x, y coordinates
        a = np.array([point_a[1], point_a[0]])  # [x, y]
        b = np.array([point_b[1], point_b[0]])  # [x, y]
        c = np.array([point_c[1], point_c[0]])  # [x, y]
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate norms
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        # Handle edge cases
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0
        
        # Calculate cosine of angle
        cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        
        # Clamp to valid range and calculate angle
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return float(np.degrees(angle))
    
    def extract_angles(self, keypoints):
        """
        Extract limb angles from keypoints
        
        Args:
            keypoints: (17, 3) array of keypoints
            
        Returns:
            List of 4 angles: [left_arm, right_arm, left_leg, right_leg]
        """
        angles = []
        
        for point_indices in self.angle_keypoints:
            i, j, k = point_indices
            angle = self.calculate_angle(keypoints[i], keypoints[j], keypoints[k])
            angles.append(angle)
        
        return angles
    
    def calculate_velocity(self, current_keypoints, previous_keypoints):
        """
        Calculate velocity features between consecutive frames
        
        Args:
            current_keypoints: (17, 3) array of current frame keypoints
            previous_keypoints: (17, 3) array of previous frame keypoints or None
            
        Returns:
            Velocity vector of length 17 (one per keypoint)
        """
        if previous_keypoints is None:
            return np.zeros(17)
        
        # Calculate Euclidean distance between corresponding keypoints
        current_coords = current_keypoints[:, :2]  # [y, x] coordinates
        previous_coords = previous_keypoints[:, :2]
        
        # Calculate velocity as L2 norm of displacement
        velocities = np.linalg.norm(current_coords - previous_coords, axis=1)
        
        return velocities
    
    def extract_features(self, keypoints, previous_keypoints=None):
        """
        Extract complete feature vector from keypoints
        
        Args:
            keypoints: (17, 3) array of keypoints
            previous_keypoints: (17, 3) array of previous keypoints or None
            
        Returns:
            Feature vector of length 21 (4 angles + 17 velocities)
        """
        # Extract angles (4 features)
        angles = self.extract_angles(keypoints)
        
        # Extract velocities (17 features)
        velocities = self.calculate_velocity(keypoints, previous_keypoints)
        
        # Combine features
        features = np.concatenate([angles, velocities])
        
        return features
    
    def validate_features(self, features):
        """
        Validate feature vector
        
        Args:
            features: Feature vector
            
        Returns:
            Boolean indicating if features are valid
        """
        if features.shape!= (Config.FEATURE_DIMENSION,):
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return False
        
        return True
    
    def normalize_features(self, features, normalization_params):
        """
        Normalize features using saved parameters
        
        Args:
            features: Feature vector or array of features
            normalization_params: Normalization parameters from training
            
        Returns:
            Normalized features
        """
        return features / (normalization_params + 1e-6)

class PersonBuffer:
    """
    Buffer to store feature sequences for individual persons
    """
    
    def __init__(self, person_id, max_length=None):
        if max_length is None:
            max_length = Config.MAX_SEQUENCE_LENGTH
        
        self.person_id = person_id
        self.max_length = max_length
        self.buffer = []
        self.feature_extractor = FeatureExtractor()
        self.previous_keypoints = None
        
    def add_keypoints(self, keypoints):
        """
        Add keypoints to buffer and extract features
        
        Args:
            keypoints: (17, 3) array of keypoints
            
        Returns:
            Boolean indicating if features were successfully added
        """
        # Extract features
        features = self.feature_extractor.extract_features(
            keypoints, self.previous_keypoints
        )
        
        # Validate features
        if not self.feature_extractor.validate_features(features):
            return False
        
        # Add to buffer
        self.buffer.append(features)
        
        # Maintain maximum length
        if len(self.buffer) > self.max_length:
            self.buffer.pop(0)
        
        # Update previous keypoints
        self.previous_keypoints = keypoints
        
        return True
    
    def get_sequence(self):
        """
        Get current sequence for prediction
        
        Returns:
            Numpy array of shape (sequence_length, feature_dim) or None
        """
        if len(self.buffer) < Config.MIN_SEQUENCE_LENGTH:
            return None
        
        return np.array(self.buffer)
    
    def is_ready_for_prediction(self):
        """
        Check if buffer has enough frames for prediction
        
        Returns:
            Boolean indicating readiness
        """
        return len(self.buffer) >= Config.MAX_SEQUENCE_LENGTH
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.previous_keypoints = None