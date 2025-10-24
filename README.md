# Sign Language Recognition

A machine learning project for American Sign Language (ASL) detection and classification using the WLASL dataset.

## Dataset

This project uses the **WLASL (Word-Level American Sign Language) v0.3** dataset, a large-scale dataset for word-level ASL recognition.

### About WLASL
- **2,000 common ASL signs/words**
- **~21,000 video instances** from multiple signers
- Multiple video sources (ASL-LEX, YouTube, etc.)
- Train/validation/test splits included
- NSLT subsets available: 100, 300, 1000, and 2000 classes

### Dataset Structure
The `data/` folder should contain:
- `WLASL_v0.3.json` - Main dataset with video metadata
- `nslt_100.json`, `nslt_300.json`, `nslt_1000.json`, `nslt_2000.json` - Configuration files for different vocabulary sizes
- `wlasl_class_list.txt` - Mapping of class indices to sign words
- `missing.txt` - List of unavailable video IDs
- `videos/` - Directory containing MP4 video files

### Dataset Download

⚠️ **Note**: The dataset is not included in this repository due to its large size (~5GB).

To use this project, you need to download the WLASL dataset:

1. Visit the official WLASL repository: [https://github.com/dxli94/WLASL](https://github.com/dxli94/WLASL)
2. Follow their instructions to download the dataset
3. Place the downloaded files in the `data/` folder following the structure above

**Citation:**
```
@inproceedings{li2020word,
  title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
  author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision},
  pages={1459--1469},
  year={2020}
}
```

## Project Structure

```
├── data/              # Dataset storage (not included - download separately)
│   ├── videos/        # MP4 video files
│   ├── WLASL_v0.3.json
│   ├── nslt_*.json
│   └── *.txt
├── src/               # Source code
│   ├── preprocess/    # Data preprocessing scripts
│   ├── models/        # Model definitions
│   └── training/      # Training scripts
├── notebooks/         # Jupyter notebooks for experiments
│   └── data_exploration.ipynb  # Dataset exploration and visualization
├── outputs/           # Model outputs and results
└── requirements.txt   # Project dependencies
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/benkedda-louai/Sign_language.git
cd Sign_language
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the WLASL dataset (see Dataset section above) and place it in the `data/` folder.

## Usage

### Data Exploration
Check out `notebooks/data_exploration.ipynb` to explore the dataset structure, statistics, and visualizations.

### Training
Coming soon...

## License

MIT License
