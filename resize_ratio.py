import os
from PIL import Image

# Set the target size for the images
TARGET_SIZE = (512, 512)

# Set the path to the folder containing the images
folder_path = '/home/matt/Desktop/lora/lorav3/lora/train/duy'

# Loop through all files in the folder
for filename in os.listdir(folder_path):
  # Check if the file is a JPEG image
  if filename.endswith(".png") or filename.endswith(".png"):
    # Open the image
    image = Image.open(os.path.join(folder_path, filename))
    # Get the original aspect ratio of the image
    width, height = image.size
    aspect_ratio = width / height
    # Calculate the new width and height based on the aspect ratio
    new_width = TARGET_SIZE[0]
    new_height = TARGET_SIZE[0] / aspect_ratio
    if new_height > TARGET_SIZE[1]:
      new_height = TARGET_SIZE[1]
      new_width = TARGET_SIZE[1] * aspect_ratio
    # Calculate the cropping coordinates
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    # Crop the image
    image = image.crop((left, top, right, bottom))
    # Resize the image
    image = image.resize(TARGET_SIZE, resample=Image.BICUBIC)
    # Save the resized image
    image.save(os.path.join(folder_path, filename))
