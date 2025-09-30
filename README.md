# Fashion MNIST Classifier
## 1. Objective
The goal of this project is to build a neural network model to classify grayscale clothing images into 10 categories (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).

This demonstrates the use of Artificial Neural Networks (ANNs) for image classification tasks.

## 2. Dataset
Fashion MNIST dataset (by Zalando), available in Keras.

60,000 training images and 10,000 test images, each of size 28×28 pixels.

Categories include shirts, trousers, coats, sneakers, etc.

Dataset is balanced across all classes.

## 3. Tools Used
TensorFlow / Keras → Model building and training.

Matplotlib & Seaborn → Visualization of accuracy/loss and confusion matrix.

Scikit-learn → Classification report and metrics.

Google Colab → Execution environment (with GPU).

##Model Architecture Explanation:
Input layer defines the image size (28×28).

Flatten layer reshapes the image into a vector of 784 features.

Dense(128, ReLU): learns patterns from the image.

Dropout(0.3): prevents overfitting by randomly deactivating neurons.

Dense(64, ReLU): adds more depth to capture features.

Dense(10, Softmax): outputs probabilities for the 10 clothing categories.

## Summary
The ANN achieved **~85–88%** accuracy on Fashion MNIST.

**Strengths**: Model learns general clothing categories well.

**Weaknesses**: Often confuses visually similar items (Shirt vs T-shirt).

**Evaluation**: Precision, Recall, and F1-score confirm balanced performance across most classes.

**Extensions**: Switching to CNN and data augmentation could push accuracy beyond 90%.

## Extensions / Improvements
Replacing ANN with **CNN (Convolutional Neural Network)** gives higher accuracy **(~92-94%)**.
we can use Data Augmentation (rotation, zoom, flip) to improve generalization.
Trying different optimizers: RMSprop, SGD with momentum.
Hyperparameter tuning (batch size, dropout rate, learning rate).
we can add Batch Normalization layers for faster convergence.
