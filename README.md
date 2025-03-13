# Digits Recognition using CNN

## Overview
This project focuses on recognizing handwritten digits using a Convolutional Neural Network (CNN). The model is trained on a dataset of handwritten digits and can accurately classify them into their respective categories (0-9).

## Dataset
The project uses the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits. The dataset is widely used for benchmarking image classification models.

## Requirements
To run this project, ensure you have the following dependencies installed:
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook (optional, for running the provided notebook)

## Project Structure
- `digits_recognition_cnn_hands_on.ipynb`: Jupyter Notebook containing the code for training and testing the CNN model.
- `README.md`: Documentation about the project.

## Steps to Run the Project
1. Install the required dependencies using:
   ```bash
   pip install tensorflow keras numpy matplotlib scikit-learn
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook digits_recognition_cnn_hands_on.ipynb
   ```
3. Run the cells step by step to train and evaluate the model.

## Model Architecture
The CNN model consists of the following layers:
- Convolutional Layers
- Max-Pooling Layers
- Flatten Layer
- Fully Connected Layers (Dense Layers)
- Output Layer with Softmax Activation

## Results
The trained model achieves high accuracy on the MNIST test set, demonstrating the effectiveness of CNNs in digit recognition.

## Future Improvements
- Experiment with different CNN architectures.
- Improve accuracy by tuning hyperparameters.
- Use data augmentation to enhance generalization.
