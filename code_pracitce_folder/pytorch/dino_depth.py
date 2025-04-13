from transformers import AutoImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

img_path = "/home/jaejun/dataset/kitti_2015/training/image_2/000104_10.png"
image = Image.open(img_path).convert("RGB")

image_processor = AutoImageProcessor.from_pretrained("facebook/dpt-dinov2-base-kitti")
model = DPTForDepthEstimation.from_pretrained("facebook/dpt-dinov2-base-kitti")
encoder = model.encoder
# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

# visualize the prediction
output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)

depth.save("depth.png")
