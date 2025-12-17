import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from collections import deque
import time
import mediapipe as mp

# Configuration
MODEL_PATH = '../model/best_model.pth'
SEQUENCE_LENGTH = 32  # Number of frames for LSTM sequence
IMG_SIZE = (224, 224)  # Input image size (adjust to your model)
CLASS_NAMES = ['baby', 'eat', 'father', 'finish', 'good', 'happy', 'hear', 'house', 'important', 'love', 'mall', 'me', 'mosque', 'mother', 'normal', 'sad', 'stop', 'thanks', 'thinking', 'worry']

# MediaPipe configuration
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Define your model architecture
class CNN_LSTM_Model(nn.Module):
    """CNN+LSTM model for sign language recognition."""
    
    def __init__(self, num_classes, hidden_size=256, num_lstm_layers=2, dropout=0.5):
        super(CNN_LSTM_Model, self).__init__()
        
        # CNN Feature Extractor (ResNet18)
        resnet = models.resnet18(pretrained=False)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_output_size = 512
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # Process each frame through CNN
        x = x.view(batch_size * seq_len, c, h, w)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(cnn_features)
        last_hidden = lstm_out[:, -1, :]
        
        # Classification
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class RealtimePredictor:
    def __init__(self, model_path, seq_len, img_size):
        self.seq_len = seq_len
        self.img_size = img_size
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
        self.hand_detected_frames = 0
        
    def load_model(self, model_path):
        """Load the trained model"""
        # Initialize model with correct architecture
        model = CNN_LSTM_Model(num_classes=len(CLASS_NAMES))
        
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
    
    def detect_hands(self, frame):
        """Detect hands in frame using MediaPipe"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(rgb_frame)
        
        # Check if hands detected
        hand_landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_landmarks_list.append(hand_landmarks)
        
        num_hands = len(hand_landmarks_list)
        return results, hand_landmarks_list, num_hands
    
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
        
    def preprocess_frame(self, frame):
        """Preprocess a single frame"""
        # Resize frame
        frame = cv2.resize(frame, self.img_size)
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        # Transpose to (C, H, W) format for PyTorch
        frame = np.transpose(frame, (2, 0, 1))
        return frame
    
    def predict(self):
        """Make prediction on buffered frames"""
        if len(self.frame_buffer) < self.seq_len:
            return None, 0.0
        
        # Stack frames into sequence
        sequence = np.array(list(self.frame_buffer))
        sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
        
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
            
            # Detect hands
            results, hand_landmarks_list, num_hands = self.detect_hands(frame)
            
            # Update hand detection status
            if hand_landmarks_list:
                self.hand_detected = True
                self.hand_detected_frames += 1
                self.no_hand_frames = 0
                
                # Draw hand landmarks
                frame = self.draw_hand_landmarks(frame, results, num_hands)
                
                # Draw bounding boxes for all detected hands
                for hand_landmarks in hand_landmarks_list:
                    x_min, y_min, x_max, y_max = self.get_hand_bbox(frame, hand_landmarks)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Preprocess and add to buffer
                processed_frame = self.preprocess_frame(frame)
                self.frame_buffer.append(processed_frame)
                
            else:
                self.no_hand_frames += 1
                self.hand_detected_frames = 0
                
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
        seq_len=SEQUENCE_LENGTH,
        img_size=IMG_SIZE
    )
    
    # Run real-time prediction
    predictor.run()