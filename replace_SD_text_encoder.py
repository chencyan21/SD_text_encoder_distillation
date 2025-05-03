import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from diffusers import StableDiffusionPipeline
import torch
from model import create_student_model_config,create_student_model
from safetensors.torch import load_file
from datasets import load_dataset
model_id = "stabilityai/stable-diffusion-2-1"
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda:6")
student_config=create_student_model_config()
stu_model=create_student_model(student_config,use_half_precision=True,device="cuda:6")
stu_model.load_state_dict(load_file("student_model_epoch43_425119aa-1fca-4f5a-964a-4b33078b652a/model.safetensors", device="cuda:6"))
pipeline.text_encoder=stu_model
pipeline = pipeline.to("cuda:6")

dataset_dict = load_dataset("captions_dataset",split="validation")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipeline(prompt).images[0]
image.save("astronaut_rides_horse_e23.png")