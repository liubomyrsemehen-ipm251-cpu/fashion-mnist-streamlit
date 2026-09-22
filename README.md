# Fashion-MNIST CNN Streamlit Classifier

A Streamlit web application for image classification using a convolutional neural network (CNN) trained on the Fashion-MNIST dataset.

## Features

- Upload an image in JPG, JPEG, or PNG format
- Display the uploaded image
- Convert the image to grayscale
- Resize the image to 28x28 pixels
- Normalize pixel values to the range 0-1
- Classify the image into one of 10 Fashion-MNIST categories
- Display the predicted class and confidence
- Visualize probabilities for all classes

## Classes

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Model

The application uses a CNN trained on Fashion-MNIST images.

- Input shape: 28x28x1
- Number of output classes: 10

## Local Installation

Install the dependencies:

    pip install -r requirements.txt

Run the application:

    streamlit run app.py

## Usage

Upload an image containing one clearly visible clothing item. The application preprocesses the image and passes it to the trained CNN for classification.

Because the model was trained on small grayscale Fashion-MNIST images, predictions on real-world photographs may be less accurate than predictions on images similar to the original dataset.
