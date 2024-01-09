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
fig, axes = plt.subplots(1, num_images_to_display, figsize=(15, 3))

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

# Show a message indicating that the image is saved
print(f"{num_images_to_display} images saved to output_images.png.")