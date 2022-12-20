import os
from PIL import Image

# Set the target size for the images
TARGET_SIZE = (512, 512)

# Set the path to the folder containing the images
folder_path = '/home/matt/Desktop/lora/lorav3/lora/train/filesaveas'

# Loop through all files in the folder
for filename in os.listdir(folder_path):
  # Check if the file is a JPEG image
  if filename.endswith(".jpg") or filename.endswith(".jpeg"):
    # Open the image
    image = Image.open(os.path.join(folder_path, filename))
    # Resize the image
    image = image.resize(TARGET_SIZE, resample=Image.BICUBIC)
    # Save the resized image
    image.save(os.path.join(folder_path, filename))
