# Face Guard

- [Link to Webapp](https://face-guard-louis-jz.streamlit.app/)
- [Link to Paper](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fabs%2F1809.00888)

## Background

Deepfake detection uses machine learning to spot manipulated videos or images, where faces or voices are altered to create false content. 
It helps combat digital deception by identifying fake media, preventing misinformation, fraud, and reputational harm, 
thus protecting the credibility of online information.

## Technical Overview

The deep fake classifier trained was based on the MesoNet architecture given the deep fake dataset by the author from this [repository](https://github.com/DariusAf/MesoNet)

Please find the data modeling code at `training_mesonet.ipynb`. This app is deployed on streamlit, with the video inference pipeline hosted on Google VertexAI servers. 

### Training Model

1. **Data Preparation**
   - Mounts Google Drive for loading datasets.
   - Defines directories for training and testing data.

2. **Data Loading**
   - Uses `ImageDataGenerator` to preprocess and rescale images for the model.
   - Loads the training and testing data with a binary classification setup (`Real` vs `Fake`).

3. **Data Visualization**
   - Visualizes a sample of images from the training dataset along with their labels (`Real` or `Fake`).

4. **MesoNet Architecture**
   - Defines the MesoNet architecture:
     - Four convolutional layers with batch normalization, followed by max-pooling.
     - Fully connected layers with dropout for regularization.
     - Final layer with sigmoid activation for binary classification.

5. **Model Compilation**
   - Compiles the model using the Adam optimizer, binary cross-entropy loss, and accuracy metrics.

6. **Training**
   - Trains the model for 15 epochs with early stopping and model checkpointing to save the best model.
   - Tracks training and validation accuracy and loss, and visualizes them after training.

7. **Model Evaluation**
   - Evaluates the model on the test dataset to calculate accuracy and loss.

### Video Inference

1. **Video Loading & Processing:**
   - The `Video` class loads video files using `imageio` and manages frame retrieval.
   - `FaceFinder` extends `Video` to detect faces in frames using `face_recognition`. It stores face coordinates and performs landmark-based analysis like center, rotation, and size.

2. **Face Detection & Alignment:**
   - `find_faces` extracts face regions from video frames, accelerating by focusing on previously detected zones or using downsampling if no face is found.
   - Aligned faces are returned using landmarks for proper rotation and zoom adjustment.

3. **Face Batch Generation:**
   - `FaceBatchGenerator` generates batches of aligned face patches for model prediction, resizing them to a target size.

4. **Prediction:**
   - `predict_faces` takes the generated batches, passes them through a classifier, and computes predictions for each face.
   - `compute_accuracy` combines face extraction and prediction to estimate the classification accuracy for a video.

### Streamlit App

1. Users upload video files via Streamlit's file_uploader.
2. Uses Google Cloud Storage (GCS) to store videos and AI Platform for the prediction model.
3. Based on the authenticity score, the UI provides feedback:
- Real if score > 0.6
- Deepfake if score < 0.4
- Uncertain for in-between scores.
