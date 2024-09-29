# ASL Alphabet Recognition

[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.3.0-red.svg)](https://keras.io/)
[![License](https://img.shields.io/github/license/Shashwat1729/ASL-Alphabet_Recognition.svg)](LICENSE)

This repository contains a Jupyter Notebook that implements an American Sign Language (ASL) alphabet recognition model using Convolutional Neural Networks (CNN). The model is trained on the Sign Language MNIST dataset to identify ASL alphabet gestures (A-Z excluding J and Z).

## Dataset

The dataset used is the [Sign Language MNIST dataset](https://www.kaggle.com/datamunge/sign-language-mnist), which consists of 27,455 training and 7,172 testing grayscale images of 28x28 pixels, each representing hand gestures for the ASL alphabet (A-Z, excluding J and Z). The dataset format is similar to the classic MNIST, with each image mapped to a label representing an ASL letter.

## Key Features

- **CNN Architecture**: A Sequential model with layers including Conv2D, MaxPooling, BatchNormalization, Dropout, and Dense layers.
- **Data Augmentation**: Techniques such as random rotation, zoom, width/height shift, and more to avoid overfitting and improve generalization.
- **Training and Evaluation**: The model is trained using Adam optimizer and categorical cross-entropy loss with callbacks like `ReduceLROnPlateau` to fine-tune learning rates.
- **Visualization**: Confusion matrices, training/validation accuracy, and loss plots for detailed performance analysis.
- **Results**: The model achieves competitive accuracy on the test set and predicts the ASL alphabet gestures effectively.

## How to Run the Notebook

1. Clone the repository:

   ```bash
   git clone https://github.com/Shashwat1729/ASL-Alphabet_Recognition.git
   cd ASL-Alphabet_Recognition
   ```

2. Run the Jupyter Notebook:

   ```bash
   jupyter notebook ASL_Alphabet_Recognition.ipynb
   ```

3. Follow the cells in the notebook to preprocess the dataset, train the model, and evaluate performance.

## Directory Structure

```
ASL-Alphabet_Recognition/
│
├── ASL_Alphabet_Recognition.ipynb   # Main notebook for model training and evaluation
├── README.md                        # Repository description and instructions
├── LICENSE                          # License information
└── requirements.txt                 # Required dependencies
```

## Model Architecture

The model architecture consists of the following layers:

- **Input Layer**: 28x28 grayscale image (1 channel).
- **Convolutional Layers**: Three Conv2D layers with ReLU activation, followed by MaxPooling and BatchNormalization.
- **Dropout Layers**: Added to reduce overfitting.
- **Dense Layers**: Fully connected layers leading to a softmax output for classification.

## Data Augmentation

To prevent overfitting, data augmentation is applied using the `ImageDataGenerator` class:

- Random rotation (10 degrees)
- Random zoom (10%)
- Random width/height shift (10%)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kaggle for providing the [Sign Language MNIST dataset](https://www.kaggle.com/datamunge/sign-language-mnist).
- TensorFlow and Keras for the deep learning framework.

---
