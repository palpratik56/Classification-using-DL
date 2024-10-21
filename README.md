## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Datasets](#datasets)
5. [Model Architectures](#model-architectures)
6. [Installation and Setup](#installation-and-setup)
7. [Usage](#usage)
8. [Results](#results)
9. [Contributing](#contributing)

## Project Overview

This project demonstrates the implementation of deep learning models for **image classification** tasks. It covers three different types of classification problems:

1. **Intel Image Classification**: Classifying natural scenes from the Intel Image dataset.
2. **Butterfly Classification**: Identifying different species of butterflies.
3. **Brain Tumor Classification**: Classifying brain MRI images into tumor or non-tumor categories.

Each classification task uses Convolutional Neural Networks (CNNs) as well as transfer learning to achieve high accuracy and performance.

## Features

- Implements three different deep learning models for different classification problems:
    1. **Intel Image Classification**: Scene classification.
    2. **Butterfly Classification**: Species identification from butterfly images.
    3. **Brain Tumor Classification**: Tumor detection from MRI images.
- Utilizes transfer learning with **pre-trained models** for better accuracy and faster convergence.
- **Data augmentation** techniques to improve model generalization.
- Performance evaluation using metrics such as accuracy, precision, and recall.

## Technologies Used

- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow/Keras
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, OpenCV
- **Modeling Techniques**: Convolutional Neural Networks (CNN), Transfer Learning (e.g., VGG16, ResNet)
- **IDE**: Jupyter Notebook

## Datasets

### 1. Intel Image Classification Dataset
- A dataset for classifying scenes into six categories: buildings, forest, glacier, mountain, sea, and street.
- [Download the dataset here](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

### 2. Butterfly Classification Dataset
- A dataset containing images of various species of butterflies.
- [Download the dataset here](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species).

### 3. Brain Tumor Classification Dataset
- A dataset of brain MRI images categorized into tumor and non-tumor.
- [Download the dataset here](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection).

## Model Architectures

Each dataset uses a specific CNN-based architecture, as described below:

### 1. Intel Image Classification
- **Architecture**: A simple CNN model with multiple convolutional layers followed by fully connected layers.
- **Activation Functions**: ReLU and Softmax for output.
- **Loss Function**: Categorical Cross-Entropy for multi-class classification.

### 2. Butterfly Classification
- **Architecture**: Utilizes **Transfer Learning** with the **VGG16** pre-trained model for feature extraction.
- **Fine-tuning**: Fine-tuned the pre-trained VGG16 model for improved performance on butterfly species.
- **Loss Function**: Categorical Cross-Entropy.

### 3. Brain Tumor Classification
- **Architecture**: Utilizes **Transfer Learning** with **ResNet50** pre-trained model for binary classification (tumor/non-tumor).
- **Loss Function**: Binary Cross-Entropy for binary classification.

## Installation and Setup

To run this project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/palpratik56/Classification-using-DL.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd Classification-using-DL
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the datasets**:
    - Download the Intel, Butterfly, and Brain Tumor datasets from the links mentioned in the [Datasets](#datasets) section.
    - Place the datasets in the appropriate folders (e.g., `data/intel/`, `data/butterfly/`, `data/brain_tumor/`).

5. **Run the Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

## Usage

Each classification task can be trained and evaluated using the corresponding Jupyter Notebook files:

- **Intel Image Classification**: Run `intel_classification.ipynb` to train and evaluate the model for scene classification.
- **Butterfly Classification**: Run `butterfly_classification.ipynb` for training and evaluation on butterfly species.
- **Brain Tumor Classification**: Run `brain_tumor_classification.ipynb` to classify brain MRI images.

Example command to train the model in the notebook:

```python
# For Intel Image Classification
model.fit(train_generator, epochs=10, validation_data=validation_generator)
```

## Results

Below are the performance metrics for each classification task:

| Task         | Model                        | Accuracy | Loss |
|--------------|------------------------------|----------|------|
| Intel Image  | Custom CNN                   | 92%      | 23%  |
| Butterfly    | VGG16 (Transfer Learning)    | 81%      | 63%  |
| Brain Tumor  | ResNet50 (Transfer Learning) | 87%      | 40%  |

Example images from the dataset with predictions:


## Contributing

Contributions are welcome! If you'd like to contribute to this project, follow these steps:

1. Fork the project.
2. Create a feature branch (`git checkout -b feature/newFeature`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/newFeature`).
5. Open a pull request.
