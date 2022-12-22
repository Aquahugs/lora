from diffusers import StableDiffusionImg2ImgPipeline
import torch
import datetime
from lora_diffusion import monkeypatch_lora, tune_lora_scale
from PIL import Image, ImageDraw, ImageFont

# Set the font and font size to use for the text labels
font = ImageFont.truetype("IBMPlexSans-Bold.ttf", 16)

init_image = Image.open("./init_image/s2r_render9.png").convert("RGB").resize((512, 512))

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    "cuda"
)

monkeypatch_lora(pipe.unet, torch.load("./output/scott_all_mjvehicle/lora_weight_e3999_s60000.pt"))

prompt = "car design concept by <zxloefoutput/scott_all_mjvehicle>, DSLR, Unrealengine"
tune_lora_scale(pipe.unet, 0.9)

torch.manual_seed(1)
# Set the initial values for strength and guidance_scale
strength = 0.1
guidance_scale = 2

# Set the increment to increase strength and guidance_scale by
strength_increment = 0.1
guidance_scale_increment = .2

# Set the number of times to iterate
num_iterations = 10

# Create an empty list to store the output images
images = []

# Draw the name of the .pt file at the top of the grid image


# Loop through the number of iterations
for i in range(num_iterations):
  # Increase strength and guidance_scale by the increment
  if strength < 0.7:
    strength += strength_increment

  guidance_scale += guidance_scale_increment

  # Run the pipeline
  image = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale).images[0]

  # Convert the image to a PIL image
  image = image.convert("RGB")

  # Add the image and the corresponding strength and guidance_scale values to the list
  images.append((image, f"strength: {strength:.2f}", f"guidance_scale: {guidance_scale:.2f}"))

# Determine the size of the grid and the size of each image
num_images = len(images)
grid_size = int(num_images**0.5)
image_size = images[0][0].size

# Create an empty image to store the grid in
grid_image = Image.new("RGB", (grid_size * image_size[0], grid_size * image_size[1]))
draw = ImageDraw.Draw(grid_image)
title_text = "lora_weight_e3999_s12000.pt"
draw.text((20, 45), title_text, font=font, fill=(0, 0, 0))

# Iterate through the images and add them to the grid
for i, (image, strength_text, guidance_scale_text) in enumerate(images):
  x = i % grid_size
  y = i // grid_size
  grid_image.paste(image, (x * image_size[0], y * image_size[1]))

  # Draw the strength and guidance scale text under the image
  draw.text((x * image_size[0], (y + 0.5) * image_size[1]), strength_text, font=font, fill=(0, 0, 0))
  draw.text((x * image_size[0], (y + 0.55) * image_size[1]), guidance_scale_text, font=font, fill=(0, 0, 0))

# Save the grid image to a file
counter = datetime.datetime.now()
grid_image.save(f"./output/images/grid{counter}.jpg", "JPEG")

