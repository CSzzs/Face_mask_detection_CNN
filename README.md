# Face Mask Detection using Convolutional Neural Networks

This project implements a CNN-based deep learning model to detect whether a person is wearing a face mask or not.

## Project Overview
The model classifies images into two categories:
- With Mask
- Without Mask

## Prerequisites
- Python 
- TensorFlow 
- Keras
- Kaggle API
- Other dependencies (Mentioned in the code file)

## Dataset
The project uses the Face Mask Detection dataset from Kaggle. To download the dataset:

1. First, you need to configure your Kaggle API credentials:
   - Go to your Kaggle account settings (https://www.kaggle.com/settings)
   - Scroll to "API" section and click "Create New API Token"
   - This will download `kaggle.json` file with your credentials

2. Extract credentials from the json file

3. The dataset will be downloaded using the following code:
```python
import opendatasets as od
# You'll be prompted to enter your Kaggle username and API key
od.download("https://www.kaggle.com/dataset-link")
```

## Model Architecture
The CNN model architecture consists of:
- Multiple Convolutional layers for feature extraction
- Max Pooling layers for dimensionality reduction
- Dense layers for classification
- Dropout layers to prevent overfitting

## How to Run
1. Clone the repository:
```bash
git clone https://github.com/CSzzs/Face_mask_detection_cnn.git
```

2. Configure Kaggle API credentials as described above

3. Run the Jupyter notebook

## Neural Network Details
The project implements a Convolutional Neural Network (CNN) with:

1. **Input Layer**
   - Accepts RGB images of specified dimensions

2. **Convolutional Layers**
   - Multiple Conv2D layers for feature extraction
   - Uses ReLU activation function
   - Filters increase progressively (32, 64, 128)

3. **Pooling Layers**
   - MaxPooling2D layers to reduce spatial dimensions
   - Helps in making the model more computationally efficient

4. **Regularization**
   - Dropout layers to prevent overfitting
   - Batch normalization for stable training

5. **Dense Layers**
   - Flattening layer to convert 2D features to 1D
   - Dense layers for final classification
   - Output layer with sigmoid activation for binary classification

## Model Training
- Uses binary cross-entropy loss function
- Adam optimizer

## Results
- Train Accuracy: 98.11%
- Test dataset Accuracy: 93.71%

## License
This project is licensed under the MIT License

## Acknowledgments
- Dataset source: Kaggle
- Inspiration: Covid-19 safety measures


## Contact
Email - chandima.senarathna97@gmail.com
Project Link: [https://github.com/CSzzs/mask-detection]
