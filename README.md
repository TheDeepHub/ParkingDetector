# Parking Slot Detector

This project demonstrates the development of a parking slot detector using a convolutional neural network (CNN), integrating the model with OpenCV for video segmentation and classification of each parking slot, optimizing the model with ONNX and C++ for performance, and deploying the model using a Flask application in a Docker container.

## Table of Contents

- [Project Overview](#project-overview)
- [1. Training the Model](#1-training-the-model)
  - [Hyperparameter Tuning with Keras Tuner](#hyperparameter-tuning-with-keras-tuner)
- [2. Integration with OpenCV in Python](#2-integration-with-opencv-in-python)
- [3. Optimization with ONNX and C++](#3-optimization-with-onnx-and-c)
- [4. Flask Application for Deployment](#4-flask-application-for-deployment)
- [Deployment with Docker](#deployment-with-docker)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to create an efficient and scalable parking slot detector. This involves several key steps:

1. **Training a CNN model** to recognize and classify parking slots as occupied or vacant.
2. **Integrating this model with OpenCV** to process video feeds and segment each parking slot.
3. **Optimizing the model** for performance using ONNX and integrating it with C++ for real-time applications.
4. **Deploying the model** in a production environment using a Flask application containerized with Docker.

## 1. Training the Model

In the `training` folder, you'll find scripts and notebooks on how to train the CNN model. This includes data preprocessing, model architecture definition, training, and evaluation.

### Hyperparameter Tuning with Keras Tuner

We use Keras Tuner to find the optimal hyperparameters for our model, ensuring the best possible accuracy. The script `hyperparameter_tuning.py` demonstrates how to set up and run hyperparameter optimization.

## 2. Integration with OpenCV in Python

After training, the model is integrated with OpenCV for real-time video processing. The `integration` folder contains Python scripts that show how to apply a mask to segment the video feed into individual parking slots and classify them using the trained model.

## 3. Optimization with ONNX and C++

For those interested in performance optimization, the `optimization` folder guides you through converting the model to the ONNX format and integrating it with C++ using OpenCV. This allows for more efficient real-time processing.

## 4. Flask Application for Deployment

The `deployment` folder contains a Flask application that serves as an interface to interact with the model. This application can receive video feed data and return predictions.

### Deployment with Docker

We also provide a Dockerfile to containerize the Flask application, ensuring easy and consistent deployment across any environment. Instructions for building and running the Docker container are included.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

