# Stable le tune
![enter image description here](https://i.ibb.co/N1tgkn2/Group-445.png)
 
 Low-rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning.
 
 ### [Paper](https://arxiv.org/abs/2106.09685)
 
TLDR: Our friends at microsoft have developed a new way to train models to understand and use new languages faster and better. It's called "Low-Rank Adaptation" (or LoRA) and it works by breaking up the computer's memory into smaller parts and only using the parts they need. This helps the computer learn the new language faster and better, even if it already knows a lot of other things. LoRA has been tested on different of contex and has been found to work really well. It can also be used using fewer parts of the computer's memory, which makes it easier and faster to learn.




##  Getting Started

### Installation

`pip install requirements.txt`

##  Preparing datasets

Store your datasets `./train` folder inside of another folder with the name of the experiment. Jpgs or Pngs are fine. 
### Formating the dataset.
Once you have your images in a folder run `resize.py` and point the `folder_path` to where you stored the images.

## Training with your own dataset
start training the stable diffusion base model.
 
`python train_lora_dreambooth.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --instance_data_dir ./train/filesaveas --instance_prompt oquboiwbcqc --output_dir ./output/filesaveas --train_text_encoder`

 - `--instance_data_dir`  the directory of your training data.



 - `--output_dir`the directory where you want to save out your models
 - `--instance_prompt` is the token word for the model.
ex. the prompt " a car design in the style of **< token here>** "  fed into a model that was trained on 20 images of bananas will try its best to give you a banana style car.
 - `--pretrained_model_name_or_path` is the stable diffusion base model
   of choice.
   
## Testing your model
Your models will be saved out in the `./output` folder inside of the folder you created that has the name of the experiment you are running.

Run `lora_infer.py` to test your model on an `init_image`

Make sure to have an image inside of the `./init_image` folder.

 Change line 18 in `lora_infer.py` to match the file directory of where your model is stored. ex. `/output/SRD_3x2_12_17_2022_A/lora_weight_e3999_s80000.pt`

Update line 81 `grid_image.save` to choose where you want your output grids to be.

## To do

 - [ ] Develop one command solution that runs multiple `.pt` files and produces multiple grids.
 - [ ] Find the magic Scott Robertson dataset configuration.
 - [ ] Moonshot: Develop a one command liner that shuffles images around folders at random and kicks off a new training job every time.
 - [ ] Figure out COG deploy logistics.
 - [ ] Find the right amount of steps in training that produces good results at a reasonable training time for the user experience.
 - [ ] Finalize Scott Robertson model.
 - [ ] Learn more about the effects of the params
 - [ ] Fly to Michigan

	