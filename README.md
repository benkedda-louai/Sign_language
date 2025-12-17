# Arabic Sign Language Recognition

A machine learning project for Arabic Sign Language (ArSL) detection and classification using computer vision and deep learning techniques.

## Dataset

This project uses the **Arabic Sign Language Dataset** from Kaggle, a comprehensive dataset for Arabic sign language recognition.

### About the Dataset

- **31 Arabic sign classes/words**
- **Compressed video files** organized by sign class
- Multiple video instances per sign
- Suitable for hand gesture recognition and classification

### Dataset Structure

The `data/` folder contains:

- `compressed videos/` - Directory containing subfolders for each sign class with MP4 video files
- `WLASL_v0.3.json` - Metadata (if applicable)
- Other configuration files and lists

### Dataset Download

⚠️ **Note**: The dataset is not included in this repository due to its size.

To use this project, download the Arabic Sign Language Dataset:

1. Visit the Kaggle dataset page: [https://www.kaggle.com/datasets/mahmoudmsaafan/arabic-sign-language-dataset](https://www.kaggle.com/datasets/mahmoudmsaafan/arabic-sign-language-dataset)
2. Download the dataset
3. Extract and place the files in the `data/compressed videos/` folder following the structure

**Citation:** Please cite the original dataset if used in research.

## Project Structure

```
├── data/                    # Dataset storage
│   ├── compressed videos/   # Video files organized by sign class
│   ├── WLASL_v0.3.json      # Metadata
│   └── other files...
├── features/                # Extracted features
│   ├── X_features.npy       # Feature arrays
│   ├── y_labels.npy         # Labels
│   └── label_encoder_classes.npy
├── model/                   # Trained models
│   ├── best_model.pth       # Best model checkpoint
│   └── other models...
├── models/                  # Additional model files
├── notebooks/               # Jupyter notebooks
│   ├── arabic-sign.ipynb    # Feature extraction
│   ├── train_model.ipynb    # Model training
│   ├── Hand_Detection_ROI_Extraction.ipynb  # Hand detection and ROI extraction
│   └── Sign_Language_CNN_LSTM_Training.ipynb  # CNN-LSTM training
├── output/                  # Processed outputs
│   └── baby_*/              # Extracted ROIs per video
└── requirements.txt         # Project dependencies
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-repo/arabic-sign-language.git
cd arabic-sign-language
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the Arabic Sign Language Dataset (see Dataset section above) and place it in the `data/` folder.

## Usage

### Pipeline Overview

This project implements a complete pipeline for Arabic sign language recognition:

1. **Feature Extraction** → Extract hand landmarks using MediaPipe
2. **Model Training** → Train ResNetLSTM model with extracted features
3. **Hand Detection & ROI Extraction** → Preprocessing pipeline using MediaPipe
4. **CNN-LSTM Training** → Deep learning model for temporal sign classification

---

## Notebook Documentation

### 1. Feature Extraction (`arabic-sign.ipynb` - First Cell)

**Purpose:** Extracts hand landmark features from Arabic sign language videos using MediaPipe for model training.

#### Technical Details

**Feature Extraction Process:**

- **MediaPipe Hands:** Detects 21 3D hand landmarks per frame
- **Video Processing:** Samples frames from videos (up to 30 frames per video)
- **Feature Vector:** 63 features per frame (21 landmarks × 3 coordinates)
- **Output:** NumPy arrays saved as `X_features.npy` and `y_labels.npy`

#### Implementation

```python
def extract_features_from_video(video_path, max_frames=30):
    # Process video frames
    # Detect hands with MediaPipe
    # Extract landmarks
    # Return feature array
```

**Key Parameters:**

- `max_frames=30`: Number of frames to sample per video
- Hand detection confidence thresholds

### 2. Model Training (`train_model.ipynb`)

**Purpose:** Trains a ResNetLSTM model for Arabic sign language classification using the extracted features.

#### Model Architecture

**ResNetLSTM Design:**

```
Input: (batch, seq_len=30, features=63)
   ↓
Feature Extractor: Linear layers (63 → 128 → 256)
   ↓
Bidirectional LSTM: 256 hidden units, 2 layers
   ↓
Classifier: Linear layers (512 → 128 → num_classes)
   ↓
Output: Class logits
```

#### Training Details

- **Dataset:** Custom dataset class for features and labels
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam with learning rate 0.001
- **Scheduler:** ReduceLROnPlateau
- **Epochs:** 200 (with early stopping)
- **Batch Size:** 16

**Training Process:**

- Loads features from `X_features.npy` and labels
- Splits data into train/validation (80/20)
- Trains the model and saves the best checkpoint

### 3. Hand Detection & ROI Extraction (`Hand_Detection_ROI_Extraction.ipynb`)

**Purpose:** Detects hands in sign language videos and extracts Regions of Interest (ROIs) for further processing.

#### Technical Architecture

**Detection Stack:**

- **MediaPipe Hands:** Real-time hand detection and landmark estimation
- **OpenCV:** Video processing and image manipulation
- **Bounding Box Extraction:** Calculates ROIs around detected hands

#### Workflow

```
Video Input → Frame Processing → Hand Detection → Landmark Extraction
                                      ↓
                          Bounding Box Calculation
                                      ↓
                              ROI Cropping → Disk Save
```

#### Key Features

- Processes videos from `data/compressed videos/`
- Extracts hand ROIs for each frame
- Saves annotated frames and individual hand images
- Supports multiple hands per frame

### 4. CNN-LSTM Training (`Sign_Language_CNN_LSTM_Training.ipynb`)

**Purpose:** Trains a CNN-LSTM model for sign language recognition using extracted ROIs.

#### Model Architecture

**CNN-LSTM Hybrid:**

```
Input: Video frames/ROIs
   ↓
CNN (ResNet): Feature extraction per frame
   ↓
LSTM: Temporal modeling across frames
   ↓
Classifier: Final prediction
```

#### Training Process

- Uses ROIs extracted from previous notebooks
- Combines spatial features (CNN) with temporal modeling (LSTM)
- Trains on sequences of frames

---

## Running the Pipeline

### Step 1: Feature Extraction

```bash
jupyter notebook notebooks/arabic-sign.ipynb
```

Run the first cell to extract hand landmarks from videos.

### Step 2: Model Training

```bash
jupyter notebook notebooks/train_model.ipynb
```

Train the ResNetLSTM model with extracted features.

### Step 3: Hand Detection & ROI Extraction

```bash
jupyter notebook notebooks/Hand_Detection_ROI_Extraction.ipynb
```

Process videos to detect hands and extract ROIs.

### Step 4: CNN-LSTM Training

```bash
jupyter notebook notebooks/Sign_Language_CNN_LSTM_Training.ipynb
```

Train the CNN-LSTM model using the ROIs.

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- MediaPipe
- NumPy
- Matplotlib
- scikit-learn

## License

MIT License
