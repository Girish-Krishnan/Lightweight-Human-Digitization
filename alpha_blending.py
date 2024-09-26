import cv2
import numpy as np
import matplotlib.pyplot as plt

def alpha_blend(img1, img2, alpha):
    """
    Alpha blending between two images based on the alpha value.
    Args:
        img1: First input image (numpy array).
        img2: Second input image (numpy array).
        alpha: Weight factor for blending. Should be between 0 and 1.
    Returns:
        Blended image.
    """
    # Ensure the images are the same size
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    
    # Perform alpha blending
    blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    return blended

def display_images(original1, original2, blended):
    """
    Displays the original and blended images side by side.
    Args:
        original1: First original input image (numpy array).
        original2: Second original input image (numpy array).
        blended: Blended image (numpy array).
    """
    # Create a figure to display the images
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes.ravel()

    ax[0].imshow(cv2.cvtColor(original1, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image 1")
    
    ax[1].imshow(cv2.cvtColor(original2, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Original Image 2")
    
    ax[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    ax[2].set_title("Blended Image")
    
    # Turn off axis labels
    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

# Load the images
image1_path = "DATASET/Subject_23/839112061696/reconstruction_images/raw_image.jpg"  # Replace with the path to your first image
image2_path = "DATASET/Subject_23/839112060979/reconstruction_images/raw_image.jpg"  # Replace with the path to your second image

img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

# Set the alpha for blending
alpha = 0.5  # You can change this value (0.0 -> only img2, 1.0 -> only img1)

# Perform image blending
blended_image = alpha_blend(img1, img2, alpha)

# Display the original and blended images
display_images(img1, img2, blended_image)
