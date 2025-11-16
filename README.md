# IMAGE-CLASSIFICATION-MODEL

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: AMIRTHAA R

*INTERN ID*: CT04DR1837

*DOMAIN*: Machine Learning

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

# DESCRIPTION ABOUT MY PROJECT 

Task 3 aims to design and train a Convolutional Neural Network (CNN) for image classification using the TensorFlow/Keras deep learning framework. CNNs are the backbone of modern computer vision applications, capable of detecting patterns, shapes, and textures in images. For this task, the CIFAR-10 dataset is used, which consists of 60,000 color images categorized into 10 classes such as airplane, dog, frog, ship, and truck. Each image is 32x32 pixels, making the dataset compact yet complex enough for meaningful CNN training.

The key steps of this task begin with loading and normalizing the dataset using TensorFlow. Normalization scales pixel values between 0 and 1, which speeds up training and improves convergence. The dataset is divided into training and test sets, ensuring that the model is evaluated on unseen data.

A CNN architecture is constructed using multiple layers of convolution, activation, pooling, flattening, and dense connections. The model begins with a Conv2D layer that extracts local features by sliding filters across the image. This is followed by MaxPooling layers that reduce spatial dimensions, improving computational efficiency and helping prevent overfitting. Additional convolution layers are added to capture deeper features. Finally, the Flatten layer converts the processed image into a single vector, which is fed into dense layers that perform final classification using softmax activation.

The model is compiled using the Adam optimizer and trained using the sparse categorical cross-entropy loss function. Training is carried out over multiple epochs, where accuracy and loss values are monitored. The model typically achieves validation accuracy in the range of 65–75%, which is expected for a simple CNN on CIFAR-10.

To analyze performance, the training history is plotted using accuracy and loss graphs. These visualizations show whether the model is learning correctly, overfitting, or underfitting. A smooth decrease in training and validation loss indicates steady learning, while increasing accuracy confirms generalization capability.

Task 3 demonstrates core deep learning concepts and provides practical experience with CNN architectures, making it an essential part of understanding machine learning for images.

# OUTPUT
