"""
Training data preparation module
"""


import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from posedetect import PoseDetector
from featureextract import FeatureExtractor
from config import Config


class DataProcessor:
    def __init__(self, show_annotations=True):
        self.pose_detector = PoseDetector()
        self.feature_extractor = FeatureExtractor()
        self.label_mapping = Config.LABEL_MAPPING
        self.show_annotations = show_annotations
        
    def process_video(self, video_path, label, max_frames=None, display_video=True):
        """
        Process a single video file with visual annotations
        
        Args:
            video_path: Path to video file
            label: Label for the video (fight/noFight)
            max_frames: Maximum frames to process
            display_video: Whether to display the video with annotations
            
        Returns:
            List of feature sequences, one per person detected
        """
        if max_frames is None:
            max_frames = Config.MAX_FRAMES_PER_VIDEO
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        # Storage for each person's features across frames
        person_buffers = []
        frame_count = 0
        
        print(f"Processing video: {os.path.basename(video_path)} - Label: {label}")
        
        # Get video properties for display
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Default fallback
        
        # Window name for display
        window_name = f"Processing: {os.path.basename(video_path)}"
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for pose detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect poses
            persons = self.pose_detector.detect_poses(rgb_frame)
            
            # Create display frame (copy of original)
            display_frame = frame.copy() if display_video else None
            
            # Ensure we have enough buffers for detected persons
            while len(person_buffers) < len(persons):
                person_buffers.append([])
            
            # Process each detected person
            for person_idx, (keypoints, bbox) in enumerate(persons):
                # Get previous keypoints for this person
                prev_keypoints = None
                if len(person_buffers[person_idx]) > 0:
                    # Get keypoints from last frame (stored with features)
                    prev_keypoints = person_buffers[person_idx][-1]['keypoints']
                
                # Extract features
                features = self.feature_extractor.extract_features(
                    keypoints, prev_keypoints
                )
                
                # Validate and store features
                if self.feature_extractor.validate_features(features):
                    person_buffers[person_idx].append({
                        'features': features,
                        'keypoints': keypoints,
                        'bbox': bbox
                    })
                
                # Draw annotations if display is enabled
                if display_video and display_frame is not None:
                    # Choose color based on label
                    color = Config.SKELETON_COLORS['fight'] if label == 'fight' else Config.SKELETON_COLORS['no_fight']
                    
                    # Draw skeleton
                    self.pose_detector.draw_skeleton(display_frame, keypoints, color)
                    
                    # Calculate and draw accurate bounding box from keypoints
                    pt1, pt2 = self.pose_detector.calculate_keypoint_bbox(keypoints, display_frame.shape)
                    
                    if pt1 is not None and pt2 is not None:
                        cv2.rectangle(display_frame, pt1, pt2, color, 2)
                        
                        # Add person ID and feature count
                        text = f"Person {person_idx}: {len(person_buffers[person_idx])} frames"
                        cv2.putText(display_frame, text, (pt1[0], pt1[1] - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Add confidence info
                        avg_conf = np.mean(keypoints[:, 2])
                        conf_text = f"Avg Conf: {avg_conf:.2f}"
                        cv2.putText(display_frame, conf_text, (pt1[0], pt2[1] + 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Display the frame with annotations
            if display_video and display_frame is not None:
                # Add video info
                info_text = f"Frame: {frame_count+1}/{max_frames} | Label: {label} | People: {len(persons)}"
                cv2.putText(display_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add instructions
                cv2.putText(display_frame, "Press 'q' to quit, 's' to skip video, SPACE to pause", 
                           (10, display_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow(window_name, display_frame)
                
                # Handle key presses
                key = cv2.waitKey(int(1000/fps)) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return []
                elif key == ord('s'):
                    break
                elif key == ord(' '):
                    # Pause functionality
                    cv2.putText(display_frame, "PAUSED - Press SPACE to continue", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow(window_name, display_frame)
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord(' '):
                            break
                        elif key == ord('q'):
                            cap.release()
                            cv2.destroyAllWindows()
                            return []
            
            frame_count += 1
            
            # Show progress in console
            if frame_count % 10 == 0:
                print(f"  Processed {frame_count} frames...")
        
        cap.release()
        if display_video:
            cv2.destroyWindow(window_name)
        
        # Extract feature sequences
        sequences = []
        for person_idx, person_buffer in enumerate(person_buffers):
            if len(person_buffer) >= Config.MIN_SEQUENCE_LENGTH:
                # Extract just the features (not keypoints/bbox)
                feature_sequence = [frame_data['features'] for frame_data in person_buffer]
                sequences.append(feature_sequence)
        
        print(f"  Found {len(person_buffers)} people, kept {len(sequences)} sequences")
        return sequences
    
    def process_dataset(self, data_path, display_videos=True):
        """
        Process entire dataset with visual annotations
        
        Args:
            data_path: Path to dataset directory
            display_videos: Whether to display videos during processing
            
        Returns:
            Tuple of (sequences, labels) where sequences is list of feature sequences
        """
        all_sequences = []
        all_labels = []
        
        print("Starting dataset processing...")
        print("Controls:")
        print("- Press 'q' to quit processing")
        print("- Press 's' to skip current video")
        print("- Press SPACE to pause/resume")
        print()
        
        # Process each label directory
        for label_dir in os.listdir(data_path):
            label_path = os.path.join(data_path, label_dir)
            
            if not os.path.isdir(label_path):
                continue
                
            if label_dir not in self.label_mapping:
                print(f"Warning: Unknown label directory '{label_dir}', skipping...")
                continue
            
            print(f"\nProcessing label: {label_dir}")
            label_value = self.label_mapping[label_dir]
            
            # Get all video files
            video_files = [f for f in os.listdir(label_path) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            
            print(f"Found {len(video_files)} videos in {label_dir}")
            
            # Process each video in the label directory
            for i, video_file in enumerate(sorted(video_files)):
                print(f"\nProcessing video {i+1}/{len(video_files)}: {video_file}")
                
                video_path = os.path.join(label_path, video_file)
                sequences = self.process_video(video_path, label_dir, display_video=display_videos)
                
                # Check if user quit
                if not sequences and display_videos:
                    print("Processing interrupted by user")
                    return all_sequences, all_labels
                
                # Add sequences and labels
                for sequence in sequences:
                    all_sequences.append(sequence)
                    all_labels.append(label_value)
        
        print(f"\nTotal sequences extracted: {len(all_sequences)}")
        return all_sequences, all_labels
    
    def prepare_training_data(self, data_path, save_data=True, display_videos=True):
        """
        Prepare training data from dataset with visual annotations
        
        Args:
            data_path: Path to dataset directory
            save_data: Whether to save processed data
            display_videos: Whether to display videos during processing
            
        Returns:
            Tuple of (X, y, normalization_params) where:
            - X: Padded sequences array
            - y: Labels array
            - normalization_params: Parameters for feature normalization
        """
        print("Starting data processing with visual annotations...")
        
        # Process dataset
        sequences, labels = self.process_dataset(data_path, display_videos=display_videos)
        
        if not sequences:
            raise RuntimeError("No valid sequences extracted from dataset")
        
        # Pad sequences to uniform length
        print("Padding sequences...")
        X = pad_sequences(
            sequences, 
            maxlen=Config.MAX_SEQUENCE_LENGTH,
            padding='post', 
            dtype='float32'
        )
        
        y = np.array(labels, dtype='float32')
        
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Fight samples: {np.sum(y == 1)}")
        print(f"No-fight samples: {np.sum(y == 0)}")
        
        # Calculate normalization parameters
        print("Calculating normalization parameters...")
        normalization_params = np.max(np.abs(X), axis=(0, 1), keepdims=True)
        
        # Normalize data
        X_normalized = X / (normalization_params + 1e-6)
        
        if save_data:
            print("Saving processed data...")
            np.save(Config.POSE_SEQUENCES_PATH, X_normalized)
            np.save(Config.POSE_LABELS_PATH, y)
            np.save(Config.NORMALIZATION_PARAMS_PATH, normalization_params)
            print("Data saved successfully!")
        
        return X_normalized, y, normalization_params
    
    def load_processed_data(self):
        """
        Load previously processed data
        
        Returns:
            Tuple of (X, y, normalization_params)
        """
        try:
            X = np.load(Config.POSE_SEQUENCES_PATH)
            y = np.load(Config.POSE_LABELS_PATH)
            normalization_params = np.load(Config.NORMALIZATION_PARAMS_PATH)
            
            print(f"Loaded data - X shape: {X.shape}, y shape: {y.shape}")
            return X, y, normalization_params
        
        except FileNotFoundError as e:
            print(f"Error loading processed data: {e}")
            print("Please run data processing first.")
            return None, None, None

if __name__ == "__main__":
    # Example usage with visual annotations
    processor = DataProcessor()
    
    # Process dataset with visual display
    X, y, norm_params = processor.prepare_training_data(
        Config.DATA_PATH, 
        display_videos=True  # Set to False to disable visual display
    )
    
    print("Data processing complete!")
    print(f"Final data shape: {X.shape}")
    print(f"Feature dimension: {X.shape[2]}")
    print(f"Max sequence length: {X.shape[1]}")


'''
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from posedetect import PoseDetector
from featureextract import FeatureExtractor
from config import Config


class DataProcessor:
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.feature_extractor = FeatureExtractor()
        self.label_mapping = Config.LABEL_MAPPING

        
    def process_video(self, video_path, label, max_frames=None):
        """
        Process a single video file
        
        Args:
            video_path: Path to video file
            label: Label for the video (fight/noFight)
            max_frames: Maximum frames to process
            
        Returns:
            List of feature sequences, one per person detected
        """
        if max_frames is None:
            max_frames = Config.MAX_FRAMES_PER_VIDEO
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        # Storage for each person's features across frames
        person_buffers = []
        frame_count = 0
        
        print(f"Processing video: {os.path.basename(video_path)}")
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect poses
            persons = self.pose_detector.detect_poses(rgb_frame)
            
            # Ensure we have enough buffers for detected persons
            while len(person_buffers) < len(persons):
                person_buffers.append([])
            
            # Process each detected person
            for person_idx, (keypoints, bbox) in enumerate(persons):
                # Get previous keypoints for this person
                prev_keypoints = None
                if len(person_buffers[person_idx]) > 0:
                    # Get keypoints from last frame (stored with features)
                    prev_keypoints = person_buffers[person_idx][-1]['keypoints']
                
                # Extract features
                features = self.feature_extractor.extract_features(
                    keypoints, prev_keypoints
                )
                
                # Validate and store features
                if self.feature_extractor.validate_features(features):
                    person_buffers[person_idx].append({
                        'features': features,
                        'keypoints': keypoints,
                        'bbox': bbox
                    })
            
            frame_count += 1
            
            # Show progress
            if frame_count % 10 == 0:
                print(f"  Processed {frame_count} frames...")
        
        cap.release()
        
        # Extract feature sequences
        sequences = []
        for person_idx, person_buffer in enumerate(person_buffers):
            if len(person_buffer) >= Config.MIN_SEQUENCE_LENGTH:
                # Extract just the features (not keypoints/bbox)
                feature_sequence = [frame_data['features'] for frame_data in person_buffer]
                sequences.append(feature_sequence)
        
        print(f"  Found {len(person_buffers)} people, kept {len(sequences)} sequences")
        return sequences
    
    def process_dataset(self, data_path):
        """
        Process entire dataset
        
        Args:
            data_path: Path to dataset directory
            
        Returns:
            Tuple of (sequences, labels) where sequences is list of feature sequences
        """
        all_sequences = []
        all_labels = []
        
        # Process each label directory
        for label_dir in os.listdir(data_path):
            label_path = os.path.join(data_path, label_dir)
            
            if not os.path.isdir(label_path):
                continue
                
            if label_dir not in self.label_mapping:
                print(f"Warning: Unknown label directory '{label_dir}', skipping...")
                continue
            
            print(f"\nProcessing label: {label_dir}")
            label_value = self.label_mapping[label_dir]
            
            # Process each video in the label directory
            for video_file in sorted(os.listdir(label_path)):
                if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    continue
                
                video_path = os.path.join(label_path, video_file)
                sequences = self.process_video(video_path, label_dir)
                
                # Add sequences and labels
                for sequence in sequences:
                    all_sequences.append(sequence)
                    all_labels.append(label_value)
        
        print(f"\nTotal sequences extracted: {len(all_sequences)}")
        return all_sequences, all_labels
    
    def prepare_training_data(self, data_path, save_data=True):
        """
        Prepare training data from dataset
        
        Args:
            data_path: Path to dataset directory
            save_data: Whether to save processed data
            
        Returns:
            Tuple of (X, y, normalization_params) where:
            - X: Padded sequences array
            - y: Labels array
            - normalization_params: Parameters for feature normalization
        """
        print("Starting data processing...")
        
        # Process dataset
        sequences, labels = self.process_dataset(data_path)
        
        if not sequences:
            raise RuntimeError("No valid sequences extracted from dataset")
        
        # Pad sequences to uniform length
        print("Padding sequences...")
        X = pad_sequences(
            sequences, 
            maxlen=Config.MAX_SEQUENCE_LENGTH,
            padding='post', 
            dtype='float32'
        )
        
        y = np.array(labels, dtype='float32')
        
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Fight samples: {np.sum(y == 1)}")
        print(f"No-fight samples: {np.sum(y == 0)}")
        
        # Calculate normalization parameters
        print("Calculating normalization parameters...")
        normalization_params = np.max(np.abs(X), axis=(0, 1), keepdims=True)
        
        # Normalize data
        X_normalized = X / (normalization_params + 1e-6)
        
        if save_data:
            print("Saving processed data...")
            np.save(Config.POSE_SEQUENCES_PATH, X_normalized)
            np.save(Config.POSE_LABELS_PATH, y)
            np.save(Config.NORMALIZATION_PARAMS_PATH, normalization_params)
            print("Data saved successfully!")
        
        return X_normalized, y, normalization_params
    
    def load_processed_data(self):
        """
        Load previously processed data
        
        Returns:
            Tuple of (X, y, normalization_params)
        """
        try:
            X = np.load(Config.POSE_SEQUENCES_PATH)
            y = np.load(Config.POSE_LABELS_PATH)
            normalization_params = np.load(Config.NORMALIZATION_PARAMS_PATH)
            
            print(f"Loaded data - X shape: {X.shape}, y shape: {y.shape}")
            return X, y, normalization_params
        
        except FileNotFoundError as e:
            print(f"Error loading processed data: {e}")
            print("Please run data processing first.")
            return None, None, None

if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    
    # Process dataset
    X, y, norm_params = processor.prepare_training_data(Config.DATA_PATH)
    
    print("Data processing complete!")
    print(f"Final data shape: {X.shape}")
    print(f"Feature dimension: {X.shape[2]}")
    print(f"Max sequence length: {X.shape[1]}")
    '''