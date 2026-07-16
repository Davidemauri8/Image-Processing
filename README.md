# Image-Processing - Leaf Detection and Classification

A MATLAB pipeline for **detecting and classifying multiple leaves in a single image**, developed for an Image Processing course project.

Given a photo (or collage) containing one or more leaves, the algorithm segments each leaf, extracts a set of geometric, texture, and color features, and classifies each leaf into one of 10 species using a K-Nearest Neighbors model — drawing a bounding box and predicted label around every detected leaf.

Authors: Davide Mauri, Tommaso Roncaglio

## Example Output

The pipeline locates every leaf in an image and labels it with its predicted species:

```
[Malva pusilla] [Erba] [Edera] [Acero] [Trifoglio]
[Rosa Irlandese] [Hoya] [Ciclamino] [Ulmus] [Quercia]
```


## Pipeline Overview

1. **Preprocessing** — convert to grayscale, apply histogram equalization and a median filter to reduce noise.
2. **Segmentation** — binarize with **Sauvola local thresholding** (chosen over global Otsu thresholding to better handle uneven lighting/background), then clean up the mask with mathematical morphology (erosion, closing, hole filling, small-object removal).
3. **Region identification** — use `regionprops` to find each connected component (leaf) and its bounding box.
4. **Feature extraction** — for each detected leaf, compute:
   - **Geometric descriptors**: Area, Perimeter, Eccentricity, Solidity, ConvexArea
   - **Texture descriptors (GLCM)**: Contrast, Correlation, Energy, Homogeneity
   - **Color descriptors**: mean and standard deviation of R, G, B channels
   
   → a 15-dimensional feature vector per leaf.
5. **Classification** — a K-Nearest Neighbors classifier (k = 1, Euclidean distance) trained on the labeled dataset predicts the species of each detected leaf.
6. **Visualization** — bounding boxes and predicted labels are drawn on the original image.

## Dataset

- **Training set**: 100 images (10 photos × 10 leaf species), photographed on a plain white background.
- **Test set**: 50 images (5 photos × 10 leaf species), photographed on varied backgrounds.
- Species: Acero, Ciclamino, Edera, Erba, Hoya, Malva pusilla, Quercia, Rosa irlandese, Trifoglio, Ulmus.

**Acquisition assumptions**: consistent lighting, photos taken from the same height (20–30 cm), captured with an iPhone 13.

**Known limitations**: small dataset size, and background color leaking into the color-based features on the test set (since test backgrounds vary while training backgrounds are uniformly white).

## Model Selection

Three classifiers were compared using MATLAB's Classification Learner and evaluated via confusion matrices:

| Model | Test Accuracy |
|---|---|
| LDA (Linear Discriminant Analysis) | 34% |
| Decision Tree (CART) | 55% |
| **KNN (k = 1)** | **58%** |

KNN was selected as the final model. Note that all three models achieve ~100% training accuracy, indicating overfitting driven by the limited dataset size and the color-feature sensitivity to background described above.

## Repository Structure

```
.
├── sauvola.m                     # Sauvola local thresholding implementation
├── train_classifier.m            # Feature extraction + KNN training on train/test sets
├── detect_and_classify_leaves.m  # Multi-leaf detection & classification pipeline
├── confmat.m                     # Confusion matrix / accuracy computation
├── show_confmat.m                # Confusion matrix visualization
└── Presentazione.pdf             # Project presentation (slides, in Italian)
```

## Usage

### 1. Train the model

Edit the `dataset_train` and `datasetPath_test` paths in `train_classifier.m` to point to your training/test folders (organized as one subfolder per species, containing `.jpg` images), then run:

```matlab
train_classifier
```

This extracts features from every image, trains the KNN model, saves it as `modKNN.mat`, and displays confusion matrices for the training and test sets.

### 2. Detect and classify leaves in a new image

Place `modKNN.mat` and an input image (e.g. `foglie.jpg`, a collage of one or more leaves) in the working directory, then run:

```matlab
detect_and_classify_leaves
```

The script segments the image, detects every leaf region, classifies each one, and displays the image with bounding boxes and predicted labels.

### Requirements

- MATLAB with the **Image Processing Toolbox** and **Statistics and Machine Learning Toolbox** (for `fitcknn`, `regionprops`, `graycomatrix`, `confusionchart`, etc.)

## Possible Improvements

- Expand the dataset (more images per class, more varied backgrounds in training) to reduce overfitting.
- Normalize or exclude color features to reduce sensitivity to background.
- Explore additional features (e.g. Hu invariant moments, Laws' texture masks) or alternative classifiers.

