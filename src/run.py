import mnist 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

# Access the training and testing datasets
train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Display a few images with labels
num_images_to_display = 5

# Create a new figure with a grid layout
fig, axes = plt.subplots(1, num_images_to_display, figsize=(15, 5))

for i in range(num_images_to_display):
    # Display each image in a subplot
    ax = plt.subplot(1, num_images_to_display, i + 1)
    ax.imshow(train_images[i], cmap='gray')
    ax.set_title(f"Label: {train_labels[i]}")
    ax.axis('off')  # Turn off axis labels

# Adjust layout for better spacing
plt.tight_layout()

# Save the figure to a file
plt.savefig("output_images.png")
plt.close()

# Show a message indicating that the image is saved
print(f"{num_images_to_display} images saved to output_images.png.")



#Now we want to Normalize and create model

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


# Reshape and normalize images
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Split the data into training and validation sets
train_images, valid_images, train_labels, valid_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)


# Model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2, padding="valid"),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    steps_per_epoch=len(train_images),
    validation_data=(valid_images, valid_labels),
    validation_steps=len(valid_images)
)



# plot the accuracy and loss

def plot_and_save_curves(history, filename="curves.png"):
    """
    Plots and saves separate loss and accuracy curves for training and validation metrics.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss and accuracy on the same figure
    
    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.savefig(filename.replace(".png", "_loss.png"))
    plt.close()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();
    
    # Save the figure
    plt.savefig(filename.replace(".png", "_accuracy.png"))
    plt.close()

plot_and_save_curves(history, filename="curves.png")


# prediction 

predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)


# Display some random test images with their predicted labels

def display_and_save_images(test_images, predicted_labels, num_samples_to_display=10, filename="predicted_images.png"):
    """
    Display and save some random test images with their predicted labels.

    """
    random_indices = np.random.choice(len(predicted_labels), num_samples_to_display, replace=False)

    for i, idx in enumerate(random_indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(test_images[idx].reshape(28, 28), cmap='summer')
        plt.title(f'Predicted: {predicted_labels[idx]}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Example usage
# Assuming 'test_images' and 'predicted_labels' are available
display_and_save_images(test_images, predicted_labels, num_samples_to_display=10, filename="predicted_images.png")


