"""
Configuration file for fight detection system
"""

class Config:
    # Model parameters
    MODEL_PATH = 'fight_detection_lstm.keras'
    NORMALIZATION_PARAMS_PATH = 'normalization_params.npy'
    POSE_SEQUENCES_PATH = 'pose_sequences.npy'
    POSE_LABELS_PATH = 'pose_labels.npy'
    
    # MoveNet model URL
    MOVENET_MODEL_URL = "https://tfhub.dev/google/movenet/multipose/lightning/1"
    
    # Sequence parameters
    MAX_SEQUENCE_LENGTH = 60
    FEATURE_DIMENSION = 21  # 4 angles + 17 velocities
    MIN_SEQUENCE_LENGTH = 10
    
    # Detection thresholds
    FIGHT_CONFIDENCE_THRESHOLD = 0.01
    POSE_CONFIDENCE_THRESHOLD = 0.2
    MIN_POSE_SCORE = 0.05
    
    # Video processing
    MAX_FRAMES_PER_VIDEO = 60
    WEBCAM_INDEX = 0
    
    # Training parameters
    TRAIN_TEST_SPLIT = 0.2
    BATCH_SIZE = 8
    EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 5
    
    # Visualization
    SKELETON_COLORS = {
        'default': (0, 255, 0),
        'fight': (0, 0, 255),
        'no_fight': (0, 255, 0),
        'detecting': (255, 255, 0)
    }
    
    # COCO pose keypoint indices
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Skeleton connections for visualization
    SKELETON_EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (0, 5), (0, 6), (5, 7), (7, 9),  # Left arm
        (6, 8), (8, 10),  # Right arm
        (5, 6), (5, 11), (6, 12),  # Torso
        (11, 12), (11, 13), (13, 15),  # Left leg
        (12, 14), (14, 16)  # Right leg
    ]
    
    # Angle calculation keypoints (for limb angles)
    ANGLE_KEYPOINTS = [
        (5, 7, 9),   # Left arm angle
        (6, 8, 10),  # Right arm angle
        (11, 13, 15), # Left leg angle
        (12, 14, 16)  # Right leg angle
    ]

    SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (8, 12), (12, 13), (13, 14),
    (0, 15), (0, 16)  # example based on COCO
]
    
    # Data directories
    DATA_PATH = 'fight-detection-surv-dataset'
    FIGHT_LABEL = 'fight'
    NO_FIGHT_LABEL = 'noFight'
    
    LABEL_MAPPING = {
        FIGHT_LABEL: 1,
        NO_FIGHT_LABEL: 0
    }