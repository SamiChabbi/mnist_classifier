"""
This script demonstrates a simple neural network training pipeline using TensorFlow and Keras on the MNIST dataset.

Functions:
    - load_and_preprocess_data(): Load and preprocess the MNIST dataset, flattening images and normalizing pixel values.
    - build_model(): Build a sequential neural network model with a Dense layer for classification.
    - compile_model(model): Compile the model with specified loss function, optimizer, and metrics.
    - train_model(model, train_data, train_labels, epochs=5): Train the model on the training data.
    - plot_curves(history): Plot training curves for loss and accuracy based on training history.
    - evaluate_model(model, test_data, test_labels): Evaluate the model on the test dataset and display accuracy and loss.
    - generate_confusion_matrix(test_labels, y_test_prediction): Generate a confusion matrix using TensorFlow.
    - plot_confusion_matrix(cm): Plot a heatmap visualization of the confusion matrix using seaborn.
    - display_example_image(x_test, test_data, test_labels): Display a random example image from the test set with the model's prediction.

Usage:
    The script loads and preprocesses data, builds and compiles a neural network, trains the model, plots training curves,
    evaluates model performance on the test set, generates a confusion matrix, and displays a random example image.

"""

import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset.

    Returns:
        flat_x_train (numpy.ndarray): Flattened training data.
        y_train (numpy.ndarray): Training labels.
        flat_x_test (numpy.ndarray): Flattened test data.
        y_test (numpy.ndarray): Test labels.
        x_test (numpy.ndarray): Original test images.
    """
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Flatten the training and testing data
    flat_x_train = x_train.reshape(x_train.shape[0], -1)
    flat_x_test = x_test.reshape(x_test.shape[0], -1)

    # Normalize pixel values to the range [0, 1]
    flat_x_train = flat_x_train.astype(float) / 255
    flat_x_test = flat_x_test.astype(float) / 255

    return flat_x_train, y_train, flat_x_test, y_test, x_test

def build_model():
    """
    Build a sequential neural network model.

    Returns:
        model (tensorflow.keras.models.Sequential): Compiled neural network model.
    """
    # Create a sequential model
    model = tf.keras.models.Sequential()

    # Add a Dense layer with 10 neurons for classification using softmax activation
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    return model

def compile_model(model):
    """
    Compile the neural network model.

    Args:
        model (tensorflow.keras.models.Sequential): Neural network model to be compiled.
    """
    # Compile the model with the loss function, optimizer, and metrics
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

def train_model(model, train_data, train_labels, epochs=5):
    """
    Train the neural network model.

    Args:
        model (tensorflow.keras.models.Sequential): Compiled neural network model.
        train_data (numpy.ndarray): Training data.
        train_labels (numpy.ndarray): Training labels.
        epochs (int): Number of training epochs.

    Returns:
        history (tensorflow.python.keras.callbacks.History): Training history.
    """
    # Callback to record information for TensorBoard
    tf_callback = tf.keras.callbacks.TensorBoard("logs/", histogram_freq=1)

    # Train the model on the data with the TensorBoard callback
    history = model.fit(train_data, train_labels, epochs=epochs, callbacks=[tf_callback])

    return history

def plot_curves(history):
    """
    Plot training curves for loss and accuracy based on training history.

    Args:
        history (tensorflow.python.keras.callbacks.History): Training history.
    """
    # Plotting the Loss curve
    plt.plot(history.history["loss"], label="loss_curve")
    plt.title("Loss")
    plt.legend()  # Add this line to display the legend
    plt.show()

    # Plotting the Accuracy curve
    plt.plot(history.history["accuracy"], label="acc_curve")
    plt.title("Accuracy")
    plt.legend()  # Add this line to display the legend
    plt.show()

def evaluate_model(model, test_data, test_labels):
    """
    Evaluate the neural network model on the test dataset and display accuracy and loss.

    Args:
        model (tensorflow.keras.models.Sequential): Trained neural network model.
        test_data (numpy.ndarray): Test data.
        test_labels (numpy.ndarray): Test labels.
    """
    # Evaluate the model on the test dataset
    evaluation = model.evaluate(test_data, test_labels)
    tested_loss, tested_accuracy = evaluation

    # Display the evaluation results
    print("Model evaluation with the test dataset:")
    print("Accuracy:", round(tested_accuracy, 4))
    print("Loss:", round(tested_loss, 4))

def generate_confusion_matrix(test_labels, y_test_prediction):
    """
    Generate a confusion matrix using TensorFlow.

    Args:
        test_labels (numpy.ndarray): True labels of the test dataset.
        y_test_prediction (numpy.ndarray): Predicted probabilities for each class.

    Returns:
        cm (tensorflow.Tensor): Confusion matrix.
    """
    # Extracting predicted labels from the model's predictions
    y_predicted_labels = [np.argmax(i) for i in y_test_prediction]

    # Generating a confusion matrix using TensorFlow
    cm = tf.math.confusion_matrix(labels=test_labels, predictions=y_predicted_labels)

    return cm

def plot_confusion_matrix(cm):
    """
    Plot a heatmap visualization of the confusion matrix using seaborn.

    Args:
        cm (tensorflow.Tensor): Confusion matrix.
    """
    # Creating a heatmap visualization of the confusion matrix using seaborn
    plt.figure(figsize=(10, 7))
    sn.heatmap(cm, annot=True, fmt='d')

    # Adding labels to the plot
    plt.xlabel('Predicted')
    plt.ylabel('Truth')

    plt.show()

def display_example_image(x_test, test_data, test_labels):
    """
    Display a random example image from the test set with the model's prediction.

    Args:
        x_test (numpy.ndarray): Original test images.
        test_data (numpy.ndarray): Test data.
        test_labels (numpy.ndarray): True labels of the test dataset.
    """
    # Generate a random index within the valid range
    random_index = np.random.randint(0, len(test_data))

    # Display the image from the test set corresponding to the random index
    plt.imshow(x_test[random_index])
    plt.show()

    # Make predictions for the randomly selected image
    random_image = np.expand_dims(test_data[random_index], axis=0)
    predictions = model.predict(random_image)
    predicted_label = np.argmax(predictions)

    # Display the model's prediction for the randomly selected image
    print("Model prediction for the randomly selected image is:", predicted_label)

if __name__ == "__main__":
    # Load and preprocess the data
    flat_x_train, y_train, flat_x_test, y_test, x_test = load_and_preprocess_data()

    # Build and compile the model
    model = build_model()
    compile_model(model)

    # Train the model
    training_history = train_model(model, flat_x_train, y_train)

    # Plot training curves
    plot_curves(training_history)

    # Evaluate the model on the test set
    evaluate_model(model, flat_x_test, y_test)

    # Make predictions on the test dataset
    y_test_prediction = model.predict(flat_x_test)

    # Generate and display the confusion matrix
    cm = generate_confusion_matrix(y_test, y_test_prediction)
    plot_confusion_matrix(cm)

    # Display an example image with a random index
    display_example_image(x_test, flat_x_test, y_test)
