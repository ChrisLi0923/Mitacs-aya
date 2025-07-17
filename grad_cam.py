import torch
import torch.nn as nn
import numpy as np
import cv2
from einops import rearrange
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

class GradCAM():
    def __init__(self, model, processor, target_layer, input_token_len, output_ids, image_mask):
        self.model = model
        self.processor = processor,
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self.input_token_len = input_token_len
        self.output_ids = output_ids
        self.target_ids = self.output_ids[self.input_token_len:]
        self.image_mask = image_mask

        # Register hooks
        target_layer.register_forward_hook(self.save_feature_maps)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output[0].detach()
        self.feature_maps.requires_grad_(True)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        

    def generate_cam_attention(self, image, inputs):
        self.model.eval()

        image_tensor = inputs["pixel_values"]
        input_ids = self.output_ids.unsqueeze(0).to(0)
        
        outputs = self.model(input_ids=input_ids, pixel_values=image_tensor)

        logits = outputs.logits[0]
        predicted_logits = logits[self.input_token_len - 1: -1]

        predicted_token_ids = torch.argmax(predicted_logits, dim=-1)

        # print(self.processor)

        print("Decoded output: ", self.processor[0].decode(predicted_token_ids, skip_special_tokens=True))
        print(len(logits))
        print(self.input_token_len)
        print(len(self.target_ids))
        target_logits = predicted_logits.gather(dim=1, index=self.target_ids.unsqueeze(1)).squeeze(1)

        # print(target_logits)
        
        target_logits = torch.sum(target_logits)
        
        target_logits.backward()
        
        print(self.feature_maps.size())
        print(self.gradients.size())

        self.feature_maps = self.feature_maps.unsqueeze(0).cpu()
        self.gradients = self.gradients.cpu()
        self.image_mask = self.image_mask.cpu()

        self.feature_maps = rearrange(self.feature_maps[:, self.image_mask, :], 'b (h w) c -> b c h w', h=13, w=13)
        self.gradients = rearrange(self.gradients[:,self.image_mask, :], 'b (h w) c -> b h w c', h=13, w=13)
        self.gradients = nn.ReLU()(self.gradients)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 1, 2])
        activation = self.feature_maps.squeeze(0)

        for i in range(activation.size(0)):
            activation[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activation, dim=0).squeeze().cpu().detach().numpy().astype(np.float32)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        threshold = 0.5
        heatmap[heatmap < threshold] = 0
        heatmap = cv2.resize(heatmap, (image.size(2), image.size(1)))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original_image = self.unprocess_image(image.squeeze().cpu().numpy())
        superimposed_img = heatmap * 0.4 + original_image
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        return heatmap, superimposed_img

    def generate_cam_vision_tower(self, image, inputs):
        self.model.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        image_tensor = inputs["pixel_values"]
        input_ids = self.output_ids.unsqueeze(0).to(0)
        
        outputs = self.model(input_ids=input_ids, pixel_values=image_tensor)

        logits = outputs.logits[0]
        predicted_logits = logits[self.input_token_len - 1: -1]

        predicted_token_ids = torch.argmax(predicted_logits, dim=-1)

        print("Decoded output: ", self.processor[0].decode(predicted_token_ids, skip_special_tokens=True))
        print(len(logits))
        # print(self.input_token_len)
        # print(len(self.target_ids))
        target_logits = predicted_logits.gather(dim=1, index=self.target_ids.unsqueeze(1)).squeeze(1)

        # print(target_logits)
        
        target_logits = torch.sum(target_logits)
        
        target_logits.backward()
        
        print(self.feature_maps.size())
        print(self.gradients.size())
        print(self.feature_maps[:200])

        self.feature_maps = self.feature_maps.unsqueeze(0).cpu()
        self.gradients = self.gradients.cpu()
        self.image_mask = self.image_mask.cpu()

        self.feature_maps = rearrange(self.feature_maps, 'b (h w) c -> b c h w', h=26, w=26)
        self.gradients = rearrange(self.gradients, 'b (h w) c -> b h w c', h=26, w=26)
        self.gradients = nn.ReLU()(self.gradients)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 1, 2])
        activation = self.feature_maps.squeeze(0)

        for i in range(activation.size(0)):
            activation[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activation, dim=0).squeeze().cpu().detach().numpy().astype(np.float32)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        threshold = 0.5
        heatmap[heatmap < threshold] = 0
        heatmap = cv2.resize(heatmap, (image.size(2), image.size(1)))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original_image = self.unprocess_image(image.squeeze().cpu().numpy())
        superimposed_img = heatmap * 0.4 + original_image
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        return heatmap, superimposed_img


    def unprocess_image(self, image):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (((image.transpose(1, 2, 0) * std) + mean) * 255).astype(np.uint8)
        return image
