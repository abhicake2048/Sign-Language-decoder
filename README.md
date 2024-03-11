
# Sign Language Decoder

This project aims to develop a deep learning model for decoding sign language gestures, allowing for the interpretation of sign language into text or spoken language. Sign language is a visual means of communication that uses hand gestures, facial expressions, and body movements to convey meaning. By leveraging computer vision and machine learning techniques, this model can recognize and interpret sign language gestures, making communication more accessible for individuals who are deaf or hard of hearing.


## Features

- Data Collection - The data used for training the project is self created for every user ,i.e., when executing the script for running the app, the user will create their own data through the use of there webcam.
- Data Preparation : The project includes scripts for collecting and preprocessing large-scale video datasets suitable for lip reading model training.
- Model Architecture : I employed advanced deep learning architectures such as convolutional neural networks (CNNs).
- Training and Evaluation : The model is trained using standard machine learning techniques and evaluated on benchmark datasets to assess its performance and accuracy.
- Deployment : Once trained, the model can be deployed as part of a real-time lip reading system, capable of transcribing spoken language from video input in various applications.


## Requirements

- Python
- Keras
- Tensorflow
- Numpy
- OpenCV

## Usage

- Clone this repository to your local machine. Install the required dependencies using ```pip install -r requirements.txt```
- Download the dataset and preprocess it using the provided scripts.
- Train the lip reading model using the prepared dataset.
- Evaluate the trained model on benchmark datasets or custom test sets.
- Deploy the model for real-time lip reading applications.
- All of the above steps, except step 1, can be done by running the single file ```app.py```
## Screenshots

![Video used for Prediction](https://drive.google.com/file/d/1FjgYKJEZUxG1ClmtOLg5b6xCuVx5mpQt/view?usp=drive_link)

![Output](https://drive.google.com/file/d/1otKOEqF46lGDkhD_wT8gOmV-iy6mZa29/view?usp=drive_link)

