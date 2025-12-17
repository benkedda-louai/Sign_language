import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import time
import mediapipe as mp

# Configuration
MODEL_PATH = '../models/best_sign_language_model_200.pth'  # Change to your model path
SEQUENCE_LENGTH = 16  # Number of frames for LSTM sequence
CLASS_NAMES = ['baby', 'eat', 'father', 'finish', 'good', 'happy', 'hear', 'house', 
               'important', 'love', 'mall', 'me', 'mosque', 'mother', 'normal', 
               'sad', 'stop', 'thanks', 'thinking', 'worry']

# MediaPipe configuration
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Define your model architecture
class ResNetLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2):
        super(ResNetLSTM, self).__init__()
        
        # Feature extractor for hand landmarks (63 features)
        self.feature_extractor = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(256, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        
        # Extract features from each frame
        x = x.view(batch_size * seq_len, features)
        x = self.feature_extractor(x)
        x = x.view(batch_size, seq_len, -1)
        
        # Process sequence with LSTM
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Use last hidden state
        
        # Classify
        x = self.fc(x)
        return x


class RealtimePredictor:
    def __init__(self, model_path, seq_len):
        self.seq_len = seq_len
        self.frame_buffer = deque(maxlen=seq_len)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self.load_model(model_path)
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Process two hands
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        
        # Hand detection status
        self.hand_detected = False
        self.no_hand_frames = 0
        
    def load_model(self, model_path):
        """Load the trained model"""
        model = ResNetLSTM(num_classes=len(CLASS_NAMES))
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        print("Model loaded successfully!")
        return model
    
    def extract_hand_landmarks(self, frame):
        """Extract hand landmarks from frame using MediaPipe"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(rgb_frame)
        
        landmarks = None
        num_hands = 0
        
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            
            # Always extract 63 features (use dominant hand or combine both)
            if num_hands >= 1:
                # For now, use the first detected hand (usually the dominant one)
                # You can modify this logic based on your needs
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                landmarks = np.array(landmarks, dtype=np.float32)
        
        return results, landmarks, num_hands
    
    def draw_hand_landmarks(self, frame, results, num_hands):
        """Draw hand landmarks on frame"""
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand label (Left or Right)
                hand_label = results.multi_handedness[idx].classification[0].label
                
                # Choose color based on hand
                if hand_label == "Left":
                    landmark_color = (0, 255, 255)  # Yellow for left
                    connection_color = (0, 200, 200)
                else:
                    landmark_color = (255, 0, 255)  # Magenta for right
                    connection_color = (200, 0, 200)
                
                # Draw landmarks with custom colors
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=connection_color, thickness=2)
                )
                
                # Add hand label text
                h, w, _ = frame.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                text_x = int(min(x_coords) * w)
                text_y = int(min(y_coords) * h) - 10
                
                cv2.putText(frame, hand_label, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, landmark_color, 2)
        
        return frame
    
    def get_hand_bbox(self, frame, hand_landmarks):
        """Get bounding box around detected hand"""
        h, w, _ = frame.shape
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        
        # Add padding
        padding = 0.1
        x_min = max(0, int((min(x_coords) - padding) * w))
        x_max = min(w, int((max(x_coords) + padding) * w))
        y_min = max(0, int((min(y_coords) - padding) * h))
        y_max = min(h, int((max(y_coords) + padding) * h))
        
        return x_min, y_min, x_max, y_max
    
    def predict(self):
        """Make prediction on buffered landmarks"""
        if len(self.frame_buffer) < self.seq_len:
            return None, 0.0
        
        # Stack landmarks into sequence (seq_len, 63)
        sequence = np.array(list(self.frame_buffer))
        sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension (1, seq_len, 63)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, class_idx = torch.max(probabilities, dim=1)
            
        return class_idx.item(), confidence.item()
    
    def run(self):
        """Run real-time prediction"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("=" * 60)
        print("Sign Language Recognition System")
        print("=" * 60)
        print(f"Model: {MODEL_PATH}")
        print(f"Classes: {len(CLASS_NAMES)}")
        print(f"Sequence Length: {SEQUENCE_LENGTH} frames")
        print("=" * 60)
        print("Show your hand to start recognition. Press 'q' to quit.")
        print("=" * 60)
        
        fps_time = time.time()
        fps_counter = 0
        fps_display = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Extract hand landmarks
            results, landmarks, num_hands = self.extract_hand_landmarks(frame)
            
            # Update hand detection status
            if landmarks is not None:
                self.hand_detected = True
                self.no_hand_frames = 0
                
                # Add landmarks to buffer
                self.frame_buffer.append(landmarks)
                
                # Draw hand landmarks
                frame = self.draw_hand_landmarks(frame, results, num_hands)
                
                # Draw bounding boxes for all detected hands
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        x_min, y_min, x_max, y_max = self.get_hand_bbox(frame, hand_landmarks)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
            else:
                self.no_hand_frames += 1
                
                # Clear buffer if no hand detected for too long
                if self.no_hand_frames > 15:
                    self.hand_detected = False
                    self.frame_buffer.clear()
            
            # Make prediction only if hand is detected
            class_idx, confidence = None, 0.0
            if self.hand_detected and len(self.frame_buffer) >= self.seq_len:
                class_idx, confidence = self.predict()
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_time > 1:
                fps_display = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Create overlay for information
            overlay = frame.copy()
            
            # Add semi-transparent background for text
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 150), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            
            # Display status and prediction
            y_offset = 35
            if self.hand_detected:
                if class_idx is not None:
                    # Show prediction with larger text
                    label = CLASS_NAMES[class_idx].upper()
                    conf_text = f"{confidence:.1%}"
                    
                    # Main prediction
                    cv2.putText(frame, f"Sign: {label}", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    
                    # Confidence
                    cv2.putText(frame, f"Confidence: {conf_text}", (10, y_offset + 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Confidence bar
                    bar_width = int(confidence * 300)
                    cv2.rectangle(frame, (10, y_offset + 55), (310, y_offset + 75), (100, 100, 100), -1)
                    cv2.rectangle(frame, (10, y_offset + 55), (10 + bar_width, y_offset + 75), (0, 255, 0), -1)
                    
                else:
                    cv2.putText(frame, "Detecting... Buffering frames", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                
                # Show number of hands detected
                hand_status = f"Status: {num_hands} HAND(S) DETECTED"
                cv2.putText(frame, hand_status, (10, y_offset + 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Status: NO HAND DETECTED", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(frame, "Please show your hand to the camera", (10, y_offset + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add FPS and buffer info in bottom left
            info_y = frame.shape[0] - 60
            cv2.putText(frame, f"FPS: {fps_display}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Buffer: {len(self.frame_buffer)}/{self.seq_len}", 
                       (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add instructions in bottom right
            instructions = "Press 'Q' to quit"
            text_size = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.putText(frame, instructions, 
                       (frame.shape[1] - text_size[0] - 10, info_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Sign Language Recognition', frame)
            
            # Press 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
        
        # Cleanup
        print("\nShutting down...")
        self.hands.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    # Initialize predictor
    predictor = RealtimePredictor(
        model_path=MODEL_PATH,
        seq_len=SEQUENCE_LENGTH
    )
    
    # Run real-time prediction
    predictor.run()