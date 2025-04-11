from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests

url = '/home/jaejun/dataset/kitti_2015/training/image_2/000104_10.png'
image = Image.open(url)

processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb8')
model = ViTModel.from_pretrained('facebook/dino-vitb8')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

print(last_hidden_states.shape)