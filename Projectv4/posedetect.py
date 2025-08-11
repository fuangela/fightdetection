"""
Unified pose detection module using MoveNet MultiPose
"""

"""
Unified pose detection module using MoveNet MultiPose with accurate bounding boxes
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from config import Config

class PoseDetector:
    def __init__(self):
        """Initialize the MoveNet MultiPose model"""
        print("Loading MoveNet MultiPose model...")
        self.model = hub.load(Config.MOVENET_MODEL_URL)
        self.movenet = self.model.signatures['serving_default']
        print("Model loaded successfully!")
    
    def detect_poses(self, frame):
        """
        Detect poses in a frame using MoveNet MultiPose
        
        Args:
            frame: RGB frame (numpy array)
            
        Returns:
            List of tuples: [(keypoints, bbox), ...] where:
            - keypoints: (17, 3) array of [y, x, confidence]
            - bbox: (4,) array of [y_min, x_min, y_max, x_max]
        """
        # Preprocess frame for MoveNet
        img = tf.image.resize_with_pad(tf.expand_dims(frame, 0), 256, 256)
        inp = tf.cast(img, tf.int32)
        
        # Run inference
        output = self.movenet(inp)['output_0'].numpy()[0]  # shape (6, 56)
        
        people = []
        for person in output:
            # Extract keypoints (first 51 values reshaped to 17x3)
            keypoints = person[:51].reshape((17, 3))
            # Extract bounding box (next 4 values)
            bbox = person[51:55]
            
            # Filter out low-confidence detections
            if np.max(keypoints[:, 2]) < Config.MIN_POSE_SCORE:
                continue
                
            people.append((keypoints, bbox))
        
        return people
    
    def keypoints_to_frame_coords(self, keypoints, frame_shape):
        """
        Convert normalized keypoints to frame coordinates
        
        Args:
            keypoints: (17, 3) array of normalized keypoints
            frame_shape: (height, width, channels) of original frame
            model_input_size: MoveNet input size (256x256)
            
        Returns:
            List of (x, y) coordinates in frame space
        """
        h, w = frame_shape[:2]
        coords = []
        for kp in keypoints:
            x_px = int(kp[1] * w)
            y_px = int(kp[0] * h)
            coords.append((x_px, y_px))
        
        return coords
    
    def calculate_keypoint_bbox(self, keypoints, frame_shape, padding=0.1):
        """
        Calculate accurate bounding box from keypoints
        
        Args:
            keypoints: (17, 3) array of keypoints [y, x, confidence]
            frame_shape: (height, width, channels) of original frame
            padding: Padding factor around keypoints
            
        Returns:
            Tuple of (pt1, pt2) for rectangle drawing, or (None, None) if no valid keypoints
        """
        # Filter out low-confidence keypoints
        valid_keypoints = keypoints[keypoints[:, 2] > Config.POSE_CONFIDENCE_THRESHOLD]
        
        if len(valid_keypoints) == 0:
            return None, None
        
        # Convert keypoints to frame coordinates
        frame_coords = self.keypoints_to_frame_coords(keypoints, frame_shape)
        
        # Filter coordinates for valid keypoints
        valid_coords = []
        for i, (x, y) in enumerate(frame_coords):
            if keypoints[i][2] > Config.POSE_CONFIDENCE_THRESHOLD:
                valid_coords.append((x, y))
        
        if not valid_coords:
            return None, None
        
        # Calculate bounding box from valid coordinates
        xs = [coord[0] for coord in valid_coords]
        ys = [coord[1] for coord in valid_coords]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Add padding
        width = max_x - min_x
        height = max_y - min_y
        
        pad_x = int(width * padding)
        pad_y = int(height * padding)
        
        # Clamp to frame boundaries
        h, w = frame_shape[:2]
        pt1 = (max(0, min_x - pad_x), max(0, min_y - pad_y))
        pt2 = (min(w, max_x + pad_x), min(h, max_y + pad_y))
        
        return pt1, pt2
    
    def draw_skeleton(self, frame, keypoints, color=(0, 255, 0), thickness=2):
        """
        Draw skeleton on frame
        
        Args:
            frame: Image frame to draw on
            keypoints: (17, 3) array of keypoints
            color: BGR color tuple
            thickness: Line thickness
        """
        
        frame_coords = self.keypoints_to_frame_coords(keypoints, frame.shape)
        
        # Draw skeleton edges
        for i, j in Config.SKELETON_EDGES:
            if (keypoints[i][2] > Config.POSE_CONFIDENCE_THRESHOLD and 
                keypoints[j][2] > Config.POSE_CONFIDENCE_THRESHOLD):
                pt1 = frame_coords[i]
                pt2 = frame_coords[j]
                cv2.line(frame, pt1, pt2, color, thickness)
        
        # Draw keypoints
        for i, (x, y) in enumerate(frame_coords):
            if keypoints[i][2] > Config.POSE_CONFIDENCE_THRESHOLD:
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    
    def draw_bounding_box(self, frame, bbox, color=(255, 0, 0), thickness=2):
        """
        Draw bounding box on frame (using MoveNet's provided bbox)
        
        Args:
            frame: Image frame to draw on
            bbox: (4,) array of [y_min, x_min, y_max, x_max] in normalized coords
            color: BGR color tuple
            thickness: Line thickness
        """
        h, w = frame.shape[:2]
        y1, x1, y2, x2 = bbox
        
        # Convert to pixel coordinates
        pt1 = (int(x1 * w), int(y1 * h))
        pt2 = (int(x2 * w), int(y2 * h))
        
        cv2.rectangle(frame, pt1, pt2, color, thickness)
        
        return pt1, pt2
    
    def draw_accurate_bounding_box(self, frame, keypoints, color=(255, 0, 0), thickness=2, padding=0.1):
        """
        Draw accurate bounding box calculated from keypoints
        
        Args:
            frame: Image frame to draw on
            keypoints: (17, 3) array of keypoints
            color: BGR color tuple
            thickness: Line thickness
            padding: Padding factor around keypoints
            
        Returns:
            Tuple of (pt1, pt2) for rectangle corners, or (None, None) if no valid keypoints
        """
        pt1, pt2 = self.calculate_keypoint_bbox(keypoints, frame.shape, padding)
        
        if pt1 is not None and pt2 is not None:
            cv2.rectangle(frame, pt1, pt2, color, thickness)
        
        return pt1, pt2

# Import OpenCV here to avoid circular imports
import cv2
'''
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from config import Config

class PoseDetector:
    def __init__(self):
        """Initialize the MoveNet MultiPose model"""
        print("Loading MoveNet MultiPose model...")
        self.model = hub.load(Config.MOVENET_MODEL_URL)
        self.movenet = self.model.signatures['serving_default']
        print("Model loaded successfully!")
    
    def detect_poses(self, frame):
        """
        Detect poses in a frame using MoveNet MultiPose
        
        Args:
            frame: RGB frame (numpy array)
            
        Returns:
            List of tuples: [(keypoints, bbox), ...] where:
            - keypoints: (17, 3) array of [y, x, confidence]
            - bbox: (4,) array of [y_min, x_min, y_max, x_max]
        """
        # Preprocess frame for MoveNet
        img = tf.image.resize_with_pad(tf.expand_dims(frame, 0), 256, 256)
        inp = tf.cast(img, tf.int32)
        
        # Run inference
        output = self.movenet(inp)['output_0'].numpy()[0]  # shape (6, 56)
        
        people = []
        for person in output:
            # Extract keypoints (first 51 values reshaped to 17x3)
            keypoints = person[:51].reshape((17, 3))
            # Extract bounding box (next 4 values)
            bbox = person[51:55]
            
            # Filter out low-confidence detections
            if np.max(keypoints[:, 2]) < Config.MIN_POSE_SCORE:
                continue
                
            people.append((keypoints, bbox))
        
        return people
    
    def keypoints_to_frame_coords(self, keypoints, frame_shape, model_input_size=256):
        """
        Convert normalized keypoints to frame coordinates
        
        Args:
            keypoints: (17, 3) array of normalized keypoints
            frame_shape: (height, width, channels) of original frame
            model_input_size: MoveNet input size (256x256)
            
        Returns:
            List of (x, y) coordinates in frame space
        """
        h, w = frame_shape[:2]
        scale = min(model_input_size / h, model_input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        pad_h = (model_input_size - new_h) // 2
        pad_w = (model_input_size - new_w) // 2
        
        frame_coords = []
        for y, x, conf in keypoints:
            # Convert from normalized coordinates to frame coordinates
            x_frame = (x * model_input_size - pad_w) / scale
            y_frame = (y * model_input_size - pad_h) / scale
            frame_coords.append((int(x_frame), int(y_frame)))
        
        return frame_coords
    
    def draw_skeleton(self, frame, keypoints, color=(0, 255, 0), thickness=2):
        """
        Draw skeleton on frame
        
        Args:
            frame: Image frame to draw on
            keypoints: (17, 3) array of keypoints
            color: BGR color tuple
            thickness: Line thickness
        """
        frame_coords = self.keypoints_to_frame_coords(keypoints, frame.shape)
        
        # Draw skeleton edges
        for i, j in Config.SKELETON_EDGES:
            if (keypoints[i][2] > Config.POSE_CONFIDENCE_THRESHOLD and 
                keypoints[j][2] > Config.POSE_CONFIDENCE_THRESHOLD):
                pt1 = frame_coords[i]
                pt2 = frame_coords[j]
                cv2.line(frame, pt1, pt2, color, thickness)
        
        # Draw keypoints
        for i, (x, y) in enumerate(frame_coords):
            if keypoints[i][2] > Config.POSE_CONFIDENCE_THRESHOLD:
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    
    def draw_bounding_box(self, frame, bbox, color=(255, 0, 0), thickness=2):
        """
        Draw bounding box on frame
        
        Args:
            frame: Image frame to draw on
            bbox: (4,) array of [y_min, x_min, y_max, x_max] in normalized coords
            color: BGR color tuple
            thickness: Line thickness
        """
        h, w = frame.shape[:2]
        y1, x1, y2, x2 = bbox
        
        # Convert to pixel coordinates
        pt1 = (int(x1 * w), int(y1 * h))
        pt2 = (int(x2 * w), int(y2 * h))
        
        cv2.rectangle(frame, pt1, pt2, color, thickness)
        
        return pt1, pt2

# Import OpenCV here to avoid circular imports
import cv2
'''