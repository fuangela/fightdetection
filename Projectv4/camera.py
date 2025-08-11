"""
Real-time fight detection using webcam
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from posedetect import PoseDetector
from featureextract import PersonBuffer
from config import Config

class FightDetector:
    def __init__(self):
        # Load trained model
        try:
            self.model = load_model(Config.MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train the model first.")
            exit(1)
        
        # Load normalization parameters
        try:
            self.normalization_params = np.load(Config.NORMALIZATION_PARAMS_PATH)
            print("Normalization parameters loaded!")
        except Exception as e:
            print(f"Error loading normalization parameters: {e}")
            exit(1)
        
        # Initialize pose detector
        self.pose_detector = PoseDetector()
        
        # Person tracking
        self.person_buffers = {}
        self.next_person_id = 0
        
        # FIXED: Store previous frame bboxes for tracking
        self.previous_bboxes = {}
        
        # Add frame counter for debugging
        self.frame_count = 0
        
    def assign_person_id(self, new_bbox, existing_bboxes, threshold=0.3):
        """Simple person tracking based on bounding box overlap"""
        best_match = None
        best_overlap = 0
        
        for person_id, old_bbox in existing_bboxes.items():
            overlap = self.calculate_bbox_overlap(new_bbox, old_bbox)
            if overlap > threshold and overlap > best_overlap:
                best_overlap = overlap
                best_match = person_id
        
        return best_match
    
    def calculate_bbox_overlap(self, bbox1, bbox2):
        """Calculate overlap between two bounding boxes"""
        y1_min, x1_min, y1_max, x1_max = bbox1
        y2_min, x2_min, y2_max, x2_max = bbox2
        
        # Calculate intersection
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def detect_fight(self, rgb_frame, display_frame):
        # Detect poses
        persons = self.pose_detector.detect_poses(rgb_frame)
        print(f"Persons detected: {len(persons)}")

        # Track current frame bboxes
        current_bboxes = {}
        fight_detected = False

        for keypoints, bbox in persons:
            # FIXED: Use previous frame's bboxes for tracking
            person_id = self.assign_person_id(bbox, self.previous_bboxes)

            if person_id is None:
                person_id = self.next_person_id
                self.next_person_id += 1
                self.person_buffers[person_id] = PersonBuffer(person_id)

            current_bboxes[person_id] = bbox

            # Add keypoints to buffer
            self.person_buffers[person_id].add_keypoints(keypoints)

            # Default color and status
            skeleton_color = (0, 255, 255)  # Yellow for detecting
            bbox_color = (139, 69, 19)
            status = f"Detecting... ({len(self.person_buffers[person_id].buffer)} frames)"

            # FIXED: Check for minimum sequence length before prediction
            if len(self.person_buffers[person_id].buffer) >= Config.MIN_SEQUENCE_LENGTH:
                sequence = self.person_buffers[person_id].get_sequence()

                if sequence is not None:
                    # Normalize
                    normalized_sequence = sequence / (self.normalization_params + 1e-6)

                    # Pad sequence to maximum length
                    if len(normalized_sequence) < Config.MAX_SEQUENCE_LENGTH:
                        padded_sequence = np.zeros((Config.MAX_SEQUENCE_LENGTH, Config.FEATURE_DIMENSION))
                        padded_sequence[:len(normalized_sequence)] = normalized_sequence
                        normalized_sequence = padded_sequence

                    # Predict
                    try:
                        prediction = self.model.predict(
                            normalized_sequence.reshape(1, Config.MAX_SEQUENCE_LENGTH, Config.FEATURE_DIMENSION),
                            verbose=0
                        )[0][0]

                        # Fight or not
                        if prediction > Config.FIGHT_CONFIDENCE_THRESHOLD:
                            fight_detected = True
                            skeleton_color = (0, 0, 255)  # Red for fight
                            status = f"FIGHT: {prediction:.3f}"
                        else:
                            skeleton_color = (0, 255, 0)  # Green for normal
                            status = f"Normal: {prediction:.3f}"
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        status = "Prediction Error"

            # Draw skeleton and bbox regardless of prediction state
            self.pose_detector.draw_skeleton(display_frame, keypoints, skeleton_color)
            
            # Calculate and draw accurate bounding box from keypoints
            pt1, pt2 = self.pose_detector.calculate_keypoint_bbox(keypoints, display_frame.shape)
            
            if pt1 is not None and pt2 is not None:
                cv2.rectangle(display_frame, pt1, pt2, skeleton_color, 2)
                
                # Draw status text
                cv2.putText(display_frame, status, (pt1[0], pt1[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, skeleton_color, 2)
                
                # Draw person ID
                cv2.putText(display_frame, f"ID: {person_id}", (pt1[0], pt2[1] + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, skeleton_color, 1)

        # FIXED: Store current bboxes for next frame and implement proper cleanup
        self.previous_bboxes = current_bboxes.copy()
        
        # Clean up unused person buffers with delay
        active_person_ids = set(current_bboxes.keys())
        for person_id in list(self.person_buffers.keys()):
            if person_id not in active_person_ids:
                # Don't immediately delete - keep for a few frames in case person reappears
                if hasattr(self.person_buffers[person_id], 'frames_missing'):
                    self.person_buffers[person_id].frames_missing += 1
                    if self.person_buffers[person_id].frames_missing > 10:  # Delete after 10 frames
                        del self.person_buffers[person_id]
                else:
                    self.person_buffers[person_id].frames_missing = 1
            else:
                # Reset missing frame counter
                if hasattr(self.person_buffers[person_id], 'frames_missing'):
                    self.person_buffers[person_id].frames_missing = 0

        return fight_detected

    def run_webcam_detection(self):
        """Run real-time detection on webcam"""
        cap = cv2.VideoCapture(Config.WEBCAM_INDEX)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        cv2.namedWindow("Fight Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Fight Detection", 1280, 720)
        
        print("Starting real-time detection. Press 'q' to quit.")
        print(f"Model will start making predictions after {Config.MIN_SEQUENCE_LENGTH} frames")
        
        while True:
            ret, frame = cap.read()  # Read frame in BGR format
            if not ret:
                break
            
            self.frame_count += 1
            
            # Convert BGR to RGB for pose detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect fight (pass BGR frame for display)
            fight_detected = self.detect_fight(rgb_frame, frame)
            
            # Add overall status
            status_text = "FIGHT DETECTED!" if fight_detected else "Normal Activity"
            status_color = (0, 0, 255) if fight_detected else (0, 255, 0)
            
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
            
            # Add frame counter and buffer info
            info_text = f"Frame: {self.frame_count} | People: {len(self.person_buffers)}"
            cv2.putText(frame, info_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Fight Detection', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FightDetector()
    detector.run_webcam_detection()
'''

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from posedetect import PoseDetector
from featureextract import PersonBuffer
from config import Config

class FightDetector:
    def __init__(self):
        # Load trained model
        try:
            self.model = load_model(Config.MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train the model first.")
            exit(1)
        
        # Load normalization parameters
        try:
            self.normalization_params = np.load(Config.NORMALIZATION_PARAMS_PATH)
            print("Normalization parameters loaded!")
        except Exception as e:
            print(f"Error loading normalization parameters: {e}")
            exit(1)
        
        # Initialize pose detector
        
        self.pose_detector = PoseDetector()
        
        # Person tracking
        self.person_buffers = {}
        self.next_person_id = 0
        
        
    def assign_person_id(self, new_bbox, existing_bboxes, threshold=0.3):
        """Simple person tracking based on bounding box overlap"""
        best_match = None
        best_overlap = 0
        
        for person_id, old_bbox in existing_bboxes.items():
            overlap = self.calculate_bbox_overlap(new_bbox, old_bbox)
            if overlap > threshold and overlap > best_overlap:
                best_overlap = overlap
                best_match = person_id
        
        return best_match
    
    def calculate_bbox_overlap(self, bbox1, bbox2):
        """Calculate overlap between two bounding boxes"""
        y1_min, x1_min, y1_max, x1_max = bbox1
        y2_min, x2_min, y2_max, x2_max = bbox2
        
        # Calculate intersection
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def detect_fight(self, rgb_frame, display_frame):

        # Detect poses
        persons = self.pose_detector.detect_poses(rgb_frame)
        print("Persons detected:", len(persons))

        # Track current frame bboxes
        current_bboxes = {}
        fight_detected = False

        for keypoints, bbox in persons:
            # Assign person ID
            person_id = self.assign_person_id(bbox, current_bboxes)

            if person_id is None:
                person_id = self.next_person_id
                self.next_person_id += 1
                self.person_buffers[person_id] = PersonBuffer(person_id)

            current_bboxes[person_id] = bbox

            # Add keypoints to buffer
            self.person_buffers[person_id].add_keypoints(keypoints)

            # Default color
            skeleton_color = (0, 255, 255)
            bbox_color = (139,69,19)
            status = "Detecting..."

            # Check if ready for prediction
            if self.person_buffers[person_id].is_ready_for_prediction():
                sequence = self.person_buffers[person_id].get_sequence()

                if sequence is not None:
                    # Normalize
                    normalized_sequence = sequence / (self.normalization_params + 1e-6)

                    # Pad
                    if len(normalized_sequence) < Config.MAX_SEQUENCE_LENGTH:
                        padded_sequence = np.zeros((Config.MAX_SEQUENCE_LENGTH, Config.FEATURE_DIMENSION))
                        padded_sequence[:len(normalized_sequence)] = normalized_sequence
                        normalized_sequence = padded_sequence

                    # Predict
                    prediction = self.model.predict(
                        normalized_sequence.reshape(1, Config.MAX_SEQUENCE_LENGTH, Config.FEATURE_DIMENSION),
                        verbose=0
                    )[0][0]

                    # Fight or not
                    if prediction > Config.FIGHT_CONFIDENCE_THRESHOLD:
                        fight_detected = True
                        skeleton_color = (0,0,255)
                        status = f"FIGHT: {prediction:.2f}"
                    else:
                        skeleton_color = (0,255,0)
                        status = f"Normal: {prediction:.2f}"

            # Draw skeleton and bbox regardless of prediction state
            self.pose_detector.draw_skeleton(display_frame, keypoints, skeleton_color)
            pt1, pt2 = self.pose_detector.draw_bounding_box(display_frame, bbox, skeleton_color)
            print("Drawing skeleton for person")


            # Draw label
            cv2.putText(display_frame, status, (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, skeleton_color, 2)

        # Clean up unused person buffers
        active_person_ids = set(current_bboxes.keys())
        for person_id in list(self.person_buffers.keys()):
            if person_id not in active_person_ids:
                del self.person_buffers[person_id]

        return fight_detected

    
    def run_webcam_detection(self):
        """Run real-time detection on webcam"""
        cap = cv2.VideoCapture(Config.WEBCAM_INDEX)
        cv2.namedWindow("Fight Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Fight Detection", 1280, 720)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting real-time detection. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()  # Read frame in BGR format
            if not ret:
                break
            
            # Convert BGR to RGB for pose detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect fight (pass BGR frame for display)
            fight_detected = self.detect_fight(rgb_frame, frame)
            
            # Add overall status
            status_text = "FIGHT DETECTED!" if fight_detected else "Normal Activity"
            status_color = (0, 0, 255) if fight_detected else (0, 255, 0)
            
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # Display frame
            cv2.imshow('Fight Detection', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FightDetector()
    detector.run_webcam_detection()

'''