#@title LOADING MODEL AND MONKEY PATCHING IT
import torch
import os
from lora_diffusion import monkeypatch_lora, tune_lora_scale
from diffusers import StableDiffusionPipeline


pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
monkeypatch_lora(pipe.unet, torch.load(os.path.join("./", "lora_weight_e1999_s4000.pt")))
monkeypatch_lora(pipe.text_encoder, torch.load(os.path.join("./", "lora_weight_e1999_s4000.text_encoder.pt")), target_replace_module=["CLIPAttention"])