# Music Genre Classification with KNN and CNN

Two baseline approaches to 10-class music genre classification on the GTZAN dataset: a K-Nearest Neighbors classifier on precomputed track-level audio descriptors, and a small convolutional neural network on precomputed Mel spectrogram images. On an 80/20 random split, the tuned KNN reached 52.50% test accuracy and the CNN reached 53.50%. Both notebooks were executed on Kaggle, and every number below is copied from their saved outputs.

## Problem

Given a 30-second music clip, predict its genre from one of ten classes: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock. Chance level is about 10%.

## Dataset

- **Source:** [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) on Kaggle (`andradaolteanu/gtzan-dataset-music-genre-classification`), which both notebooks download with `kagglehub`.
- **Contents used:**
  - `Data/features_30_sec.csv`: one row per track (1,000 rows, 60 columns) of precomputed descriptors. Used by the KNN notebook.
  - `Data/images_original/<genre>/*.png`: one Mel spectrogram image per track. Used by the CNN notebook, which loaded 999 images (jazz has 99).
  - `Data/genres_original/` (raw `.wav` audio) and `Data/features_3_sec.csv` are present but not used.
- **Not included in this repository.** Download it yourself (see [How to run](#how-to-run)).

## Approach

### KNN (`KNN.ipynb`)

- **Features:** 10 columns from `features_30_sec.csv`: `chroma_stft_mean`, `rms_mean`, `spectral_centroid_mean`, `spectral_bandwidth_mean`, `rolloff_mean`, `zero_crossing_rate_mean`, `harmony_mean`, `perceptr_mean`, `tempo`, `mfcc1_mean`. No features are extracted from audio in the notebook.
- **Split:** `train_test_split(test_size=0.2, random_state=42)`, giving 800 training and 200 test tracks. The split is not stratified.
- **Experiments:**
  1. KNN with k=5 on unscaled features.
  2. The same model after `StandardScaler`, fitted on the training set only.
  3. `GridSearchCV` (5-fold CV on the training set) over `n_neighbors` {3, 5, 7, 9, 11, 13}, `weights` {uniform, distance} and `metric` {euclidean, manhattan, minkowski}. Best: `{'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'distance'}`.
  4. A second grid over `n_neighbors` {7, 15, 21, 25} and `metric` {manhattan, cosine, chebyshev} picked the same configuration.
  5. PCA with 10 components on the scaled features, followed by KNN with default parameters (k=5).

### CNN (`CNN.ipynb`)

- **Input:** the dataset's Mel spectrogram PNGs, resized to 128x128 RGB and scaled to [0, 1].
- **Split:** `train_test_split(test_size=0.2, random_state=42)` over the 999 images, giving 799 training and 200 test images.
- **Architecture (Keras):** Conv2D(32, 3x3, ReLU), MaxPool(2x2), Conv2D(64, 3x3, ReLU), MaxPool(2x2), Flatten, Dense(128, ReLU), Dropout(0.3), Dense(10, softmax).
- **Training:** Adam optimizer, categorical cross-entropy loss, 50 epochs, batch size 32. No augmentation, early stopping or checkpointing.
- `CNN.ipynb` defines a `safe_load_wav` helper (scipy + librosa) for raw audio, but nothing calls it.

## Results

All values are copied from the executed notebook outputs. Each test split has 200 items. The KNN and CNN splits use the same seed but draw from slightly different pools (1,000 tracks and 999 images), so the two test sets are not identical.

| Model | Evaluation | Accuracy |
|---|---|---|
| KNN, k=5, unscaled features | Test (200) | 34.00% |
| KNN, k=5, standardized features | Test (200) | 50.00% |
| KNN, tuned (manhattan, k=7, distance weights) | Test (200) | **52.50%** |
| KNN, tuned (same configuration) | 5-fold CV mean on training set (800) | 59.00% |
| PCA (10 components) + KNN, k=5 | Test (200) | 50.00% |
| CNN, final epoch (50) | Test (200) | **53.50%** |

Tuned KNN on the test set: macro F1 0.52, weighted F1 0.51. The best per-class F1 scores were classical (0.87), jazz (0.70) and metal (0.68). The worst were rock (0.26), hiphop (0.34) and reggae (0.39).

CNN: training accuracy was about 0.99 to 1.00 over the last epochs, while test accuracy stayed between 0.50 and 0.585 from epoch 5 onward. The model overfits heavily.

## Limitations

- **Small dataset and a single split.** There are about 1,000 tracks and each test set has 200 items, so one test example is 0.5 percentage points. No repeated splits or confidence intervals were computed, and the split is not stratified (per-class KNN test support ranges from 13 to 27).
- **The test set doubles as the CNN validation set.** `model.fit(..., validation_data=(X_test, y_test))` monitors the test set during training. The reported number is from the final epoch, not one selected on the test set, but there is no separate validation set.
- **Model choice was made on the test set.** Several KNN variants were compared on the same test split. The 59.00% figure is the cross-validation score on the training data, not test accuracy, even though the notebook labels it "Accuracy of Best KNN model".
- **The PCA step reduces nothing.** It keeps 10 components of 10 features, so it is only a rotation. It also reuses an unfitted default `KNeighborsClassifier` (k=5), not the tuned model.
- **Cell execution order.** In `KNN.ipynb`, a cell that uses `PCA` comes before the cell that imports it (execution counts 16 and 17). Run top to bottom in a fresh kernel, that cell fails until the import cell has run. The last cell of `KNN.ipynb` is empty and was never executed.
- **Results are not exactly reproducible.** No TensorFlow seed is set, so CNN results will vary between runs.
- **No track-level leakage was found, but the dataset has known problems.** Both models use one sample per 30-second track (the 3-second segment features are not used), so segments of the same track cannot end up in both train and test. GTZAN itself has known issues, such as repeated artists and duplicated or mislabeled clips. The random split does not account for these, so the scores may be optimistic.
- **The CNN is a basic baseline.** It has two convolutional blocks, no augmentation and no regularization beyond one dropout layer.

## How to run

Tested setup: Python 3.11 (the notebooks ran on Kaggle with Python 3.11 and TensorFlow 2.18.0).

```bash
git clone https://github.com/Akshatb848/Music-Genre-Classification-USING-KNN-and-CNN.git
cd Music-Genre-Classification-USING-KNN-and-CNN
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

### Getting the data

- **Option A, run on Kaggle (easiest).** Open a notebook on Kaggle, attach the dataset `andradaolteanu/gtzan-dataset-music-genre-classification`, and upload or paste the notebooks. The hardcoded paths under `/kaggle/input/...` work as they are.
- **Option B, run locally.**
  1. Set up Kaggle API credentials: either `~/.kaggle/kaggle.json` or the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables. Do not commit them.
  2. Run the first `kagglehub.dataset_download(...)` cell. It downloads the dataset to a local cache and prints the path.
  3. Both notebooks hardcode `/kaggle/input/gtzan-dataset-music-genre-classification`. Replace that path with the printed one: `dataset_path` in `KNN.ipynb` and `image_dir` in `CNN.ipynb`. The expected layout under that root is `Data/features_30_sec.csv` and `Data/images_original/<genre>/*.png`.

The first two cells of `KNN.ipynb` run `!pip install ...`. These are not needed once `requirements.txt` is installed.

## Project structure

```
.
├── KNN.ipynb          # Feature-based KNN: baseline, scaling, GridSearchCV tuning, PCA variant
├── CNN.ipynb          # Spectrogram-image CNN (Keras)
├── requirements.txt   # Python dependencies (versions from the executed Kaggle environment)
└── README.md
```
