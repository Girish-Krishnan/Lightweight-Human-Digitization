import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt

def match_histograms(source_image, reference_image):
    """
    Matches the histogram of the source image to that of the reference image.
    Args:
        source_image: Image to be adjusted (numpy array).
        reference_image: Image whose histogram will be matched (numpy array).
    Returns:
        The source image with its histogram matched to the reference image.
    """
    matched = exposure.match_histograms(source_image, reference_image)
    return matched

def display_images_comparison(original, matched):
    """
    Displays the original and matched images side by side.
    Args:
        original: Original input image (numpy array).
        matched: Histogram-matched image (numpy array).
    """

    cv2.imshow("Original", original)
    cv2.imshow("Matched", matched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load the images
source_image_path = "DATASET/Subject_23/839112061696/reconstruction_images/raw_image.jpg"  # Replace with the path to your source image
reference_image_path = "DATASET/Subject_23/839112060979/reconstruction_images/raw_image.jpg"  # Replace with the path to your reference image

source_image = cv2.imread(source_image_path)
reference_image = cv2.imread(reference_image_path)

# Perform histogram matching
matched_image = match_histograms(source_image, reference_image)

# Display the result
display_images_comparison(source_image, matched_image)
