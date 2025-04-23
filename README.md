# 🎵 Music Genre Classification with KNN & CNN

This project aims to classify music into genres using two different machine learning approaches: **K-Nearest Neighbors (KNN)** and **Convolutional Neural Networks (CNN)**. The dataset used is the popular **GTZAN Music Genre Dataset**, containing 1000 audio tracks categorized into 10 genres.

---

## 📁 Dataset

**GTZAN Genre Collection**  
- 10 genres: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`
- Each genre contains 100 audio files in `.wav` format (~30 seconds)

You can download the dataset from [GTZAN Dataset on Kaggle](https://www.kaggle.com/datasets/chrisfilo/gtzan-dataset-music-genre-classification).

---

## 🧠 Models Used

### 🔹 K-Nearest Neighbors (KNN)
- Audio files are converted to MFCC (Mel-frequency cepstral coefficients)
- Each audio file is represented by a fixed-length feature vector
- Hyperparameters optimized using GridSearchCV

**Best Hyperparameters**:
```python
{
  "metric": "manhattan",
  "n_neighbors": 7,
  "weights": "distance"
}
