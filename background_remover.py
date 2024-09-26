import argparse
from rembg import remove
from PIL import Image
import io

def remove_background(image_path):
    # Read the image file
    with open(image_path, 'rb') as img_file:
        input_image = img_file.read()

    # Remove the background
    output_image = remove(input_image)

    # Load the output image into PIL for saving
    image = Image.open(io.BytesIO(output_image))
    image = image.convert("RGBA")  # Ensure it has an alpha channel

    # Save the image with a transparent background
    output_path = image_path.replace('.jpg', '_no_bg.png')
    image.save(output_path)

    print(f"Background removed. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove background from a JPG image.")
    parser.add_argument("image_path", type=str, help="Path to the image file (JPG format)")

    args = parser.parse_args()

    remove_background(args.image_path)
