import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from grad_cam import GradCAM, SmoothGradCAM
from utils import load_model

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model, processor = load_model()

conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "Is giraffe on this image? Provide only yes no response"},
          {"type": "image"},
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image_file = "/h/mikl123/Mitacs-aya/images/giraffe.png"
raw_image = Image.open(image_file).convert("RGB")
raw_image = raw_image.resize((364, 364))

inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

input_ids = inputs["input_ids"][0]
output_ids = output[0]
image_mask = output_ids == 255036
print("Image mask ", sum(image_mask))
print("Len output ", len(image_mask))
print("Len input ", len(input_ids))
transform = transforms.ToTensor()
image_tensor = transform(raw_image)

# print(output_ids)
# print(model)
# print(image_mask)

def make_attention_grad_cam():
    grid_cols = 8
    grid_rows = 4
    img_h, img_w = 364, 364

    layers = model.language_model.model.layers
    print(model)
    all_images = []
    for i in range(32):
        target_layer = layers[i].input_layernorm
        gradcam = SmoothGradCAM(
            model,
            processor,
            target_layer,
            num_samples = 1, noise_std = 0,
            input_token_len=len(input_ids),
            output_ids=output_ids,
            image_mask=image_mask
        )
        _, overlay_img = gradcam.generate_smooth_cam(image_tensor, inputs, vision_tower = False)
        if overlay_img.shape[:2] != (img_h, img_w):
            overlay_img = cv2.resize(overlay_img, (img_w, img_h))
        all_images.append(overlay_img)

    grid_img = np.zeros((img_h * grid_rows, img_w * grid_cols, 3), dtype=np.uint8)
    for idx, img in enumerate(all_images):
        row = idx // grid_cols
        col = idx % grid_cols

        y1 = row * img_h
        y2 = y1 + img_h
        x1 = col * img_w
        x2 = x1 + img_w

        grid_img[y1:y2, x1:x2, :] = img

    cv2.imwrite("/h/mikl123/Mitacs-aya/results/irafe_att.jpg", grid_img)

def make_vision_grad_cam():
    grid_cols = 9
    grid_rows = 3
    img_h, img_w = 364, 364

    layers = model.vision_tower.vision_model.encoder.layers
    all_images = []
    for i in range(27):
        target_layer = layers[i].layer_norm2
        gradcam = SmoothGradCAM(
            model,
            processor,
            target_layer,
            num_samples = 1, noise_std = 0,
            input_token_len=len(input_ids),
            output_ids=output_ids,
            image_mask=image_mask
        )
        heatmap, overlay_img = gradcam.generate_smooth_cam(image_tensor, inputs, vision_tower=True)
        if overlay_img.shape[:2] != (img_h, img_w):
            overlay_img = cv2.resize(overlay_img, (img_w, img_h))
        all_images.append(overlay_img)

    grid_img = np.zeros((img_h * grid_rows, img_w * grid_cols, 3), dtype=np.uint8)

    for idx, img in enumerate(all_images):
        row = idx // grid_cols
        col = idx % grid_cols

        y1 = row * img_h
        y2 = y1 + img_h
        x1 = col * img_w
        x2 = x1 + img_w

        grid_img[y1:y2, x1:x2, :] = img

    cv2.imwrite("/h/mikl123/Mitacs-aya/results/irafe_vision.jpg", grid_img)

def make_backprop(): 
    all_images = []
    gradcam = GradCAM(
        model,
        processor,
        None,
        input_token_len=len(input_ids),
        output_ids=output_ids,
        image_mask=image_mask
    )
    heatmap = gradcam.generate_cam_input(image_tensor, inputs) 

    heatmap_img = Image.fromarray(heatmap)

    heatmap_img.save("/h/mikl123/Mitacs-aya/results/heatmap_giraffe.png")


make_attention_grad_cam()
make_vision_grad_cam()
# make_backprop()