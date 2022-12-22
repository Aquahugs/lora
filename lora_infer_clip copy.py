from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
from lora_diffusion import patch_pipe, tune_lora_scale

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    "cuda:0"
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

prompt = "<zxloefoutput/scott_all_mjvehicle> black sports car parked next to a parking meter, by <zxloefoutput/scott_all_mjvehicle> "

patch_pipe(pipe, "./output/scott_all_mjvehicle/lora_weight_e3999_s60000.pt",prompt, patch_text=True)
tune_lora_scale(pipe.unet, 1.00)

torch.manual_seed(12)
image = pipe(prompt, num_inference_steps=100, guidance_scale=3).images[0]
image.save("./output/images/scott_3.jpg")
image  # Wow ok, now I might have to deal with a lawsuite for this.