import cv2
import numpy as np
import matplotlib.pyplot as plt

def shades_of_gray(image, power=6, norm=6):
    """
    Apply Shades of Gray photometric calibration for color constancy.
    Args:
        image: Input image (numpy array).
        power: Power used to compute the Minkowski norm (default=6).
        norm: Type of norm used for scaling (default=6 for Minkowski norm).
    Returns:
        Calibrated image (numpy array).
    """
    # Separate the color channels
    red_channel = image[:, :, 2].astype(np.float32)
    green_channel = image[:, :, 1].astype(np.float32)
    blue_channel = image[:, :, 0].astype(np.float32)

    # Compute the Minkowski norm for each color channel
    red_norm = np.power(np.mean(np.power(red_channel, power)), 1 / power)
    green_norm = np.power(np.mean(np.power(green_channel, power)), 1 / power)
    blue_norm = np.power(np.mean(np.power(blue_channel, power)), 1 / power)

    # Compute the scaling factors to normalize the channels
    scale = (red_norm + green_norm + blue_norm) / 3.0
    red_scale = scale / red_norm
    green_scale = scale / green_norm
    blue_scale = scale / blue_norm

    # Apply the scaling to each channel
    red_channel *= red_scale
    green_channel *= green_scale
    blue_channel *= blue_scale

    # Merge the channels back together and clip values to the valid range
    calibrated_image = np.clip(cv2.merge([blue_channel, green_channel, red_channel]), 0, 255).astype(np.uint8)
    
    return calibrated_image

def display_images(original, calibrated):
    """
    Displays the original and photometrically calibrated images side by side.
    Args:
        original: Original input image (numpy array).
        calibrated: Calibrated image (numpy array).
    """
    # Create a figure to display the original and calibrated images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes.ravel()

    ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image")
    
    ax[1].imshow(cv2.cvtColor(calibrated, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Shades of Gray Calibrated Image")
    
    # Turn off axis labels
    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

# Load the image
image_path = "DATASET/Subject_23/839112060979/reconstruction_images/image_no_bg.png"  # Replace with the path to your input image
image = cv2.imread(image_path)

# Apply photometric calibration (Shades of Gray)
image_calibrated = shades_of_gray(image)

# Display the original and calibrated images
display_images(image, image_calibrated)
