import torch
import torch.nn as nn
import numpy as np
import cv2
from einops import rearrange
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from tqdm import tqdm


from PIL import Image
import numpy as np

def save_tensor_image(tensor, filename="output.png", unnormalize=False, mean=None, std=None):
    tensor = tensor.detach().cpu().squeeze()

    if unnormalize and mean and std:
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)

    array = tensor.numpy()
    array = np.clip(array, 0, 1)
    array = (array * 255).astype(np.uint8)

    array = np.transpose(array, (1, 2, 0))

    Image.fromarray(array).save(filename)


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
        if target_layer is not None:
            target_layer.register_forward_hook(self.save_feature_maps)
            target_layer.register_full_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output[0].detach()
        self.feature_maps.requires_grad_(True)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_cam(self, image, vision_tower = False, smooth = False, logs = False):
        self.model.eval()
        image_tensor = image
        input_ids = self.output_ids.unsqueeze(0).to(0)
        
        outputs = self.model(input_ids=input_ids, pixel_values=image_tensor)
        logits = outputs.logits[0]
        predicted_logits = torch.softmax(logits[self.input_token_len - 1: -1], dim = -1)

        if logs:
            predicted_token_ids = torch.argmax(predicted_logits, dim=-1)
            print("Decoded output: ", self.processor[0].decode(predicted_token_ids, skip_special_tokens=False))
            print("Original output: ", self.processor[0].decode(self.target_ids, skip_special_tokens=False))
            print("Predicted ids ",predicted_logits.argmax(dim=-1))
            print("Target ids ",self.target_ids)

        predicted_logits = predicted_logits[1:-2]
        target_ids = self.target_ids[1:-2]
        target_logits = predicted_logits.gather(dim=1, index=target_ids.unsqueeze(1)).squeeze(1)
        if not smooth:
            assert all(predicted_logits.argmax(dim=-1) == target_ids), 'ids is not the same'
            print("passed ")
        
        target_logits = torch.sum(target_logits)
        self.model.zero_grad()
        target_logits.backward()
        
        if logs:
            print("Feature map shape", self.feature_maps.size())
            print("Gradient shape", self.gradients.size())

        self.feature_maps = self.feature_maps.unsqueeze(0).cpu()
        self.gradients = self.gradients.cpu()
        self.image_mask = self.image_mask.cpu()

        if not vision_tower:
            self.feature_maps = rearrange(self.feature_maps[:, self.image_mask, :], 'b (h w) c -> b c h w', h=13, w=13)
            self.gradients = rearrange(self.gradients[:,self.image_mask, :], 'b (h w) c -> b h w c', h=13, w=13)
        else:
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
        heatmap = cv2.resize(heatmap, (image.size(3), image.size(2)))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        print("heatmap ", heatmap.shape)
        original_image = self.unprocess_image(image.squeeze().cpu().numpy())
        superimposed_img = heatmap * 0.4 + original_image
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        return heatmap, superimposed_img
    
    def generate_cam_input(self, image, inputs): 
        self.model.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        image_tensor = inputs["pixel_values"].requires_grad_(True)
        input_ids = self.output_ids.unsqueeze(0).to(0)
        
        outputs = self.model(input_ids=input_ids, pixel_values=image_tensor) 

        logits = outputs.logits[0]
        predicted_logits = logits[self.input_token_len - 1: -1]
        predicted_token_ids = torch.argmax(predicted_logits, dim=-1)
        print("Decoded output: ", self.processor[0].decode(predicted_token_ids, skip_special_tokens=True))
        print(len(logits))

        target_logits = predicted_logits.gather(dim=1, index=self.target_ids.unsqueeze(1)).squeeze(1)
        assert all(predicted_logits.argmax(dim=-1) == self.target_ids), 'ids is not the same'
        target_logits = torch.sum(target_logits)

        self.model.zero_grad()
        
        if image_tensor.grad is not None:
            image_tensor.grad.zero_()

        target_logits.backward()
        image_grad = image_tensor.grad.clone().detach() 
        # image_grad = image_grad * inputs["pixel_values"].clone().detach() 

        image_grad = image_grad.squeeze(0) 

        image_grad = torch.abs(image_grad)
        image_grad = image_grad.permute(1, 2, 0)  # PyTorch step
        image_grad = image_grad.mean(axis = -1)
        grad_np = image_grad.cpu().numpy().astype(np.float64)
        print("Any NaNs:", np.isnan(grad_np).any())
        print("Any Infs:", np.isinf(grad_np).any())
        print("Min:", np.nanmin(grad_np), "Max:", np.nanmax(grad_np))

        grad_np = self.normalize_grad_for_display(grad_np)
        print(grad_np.shape)
        # print(grad_np)

        return grad_np
    
    def normalize_grad_for_display(self, grad_np, lower=2, upper=98):
        low_val = np.percentile(grad_np, lower)
        high_val = np.percentile(grad_np, upper)
        grad_np = np.clip(grad_np, low_val, high_val)

        mean = grad_np.mean()
        std = grad_np.std()
        grad_np = (grad_np - mean) / (std + 1e-5)
        grad_np = (grad_np - grad_np.min()) / (grad_np.max() - grad_np.min())
        grad_np = (grad_np * 255).astype(np.uint8)

        return grad_np

    def unprocess_image(self, image):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (((image.transpose(1, 2, 0) * std) + mean) * 255).astype(np.uint8)
        return image

class SmoothGradCAM(GradCAM):
    def __init__(self, model, processor, target_layer, input_token_len, output_ids, image_mask, num_samples=1, noise_std=0.1):
        super().__init__(model, processor, target_layer, input_token_len, output_ids, image_mask)
        self.num_samples = num_samples
        self.noise_std = noise_std

    def add_noise(self, tensor, noise_std):
        noise = torch.randn_like(tensor) * noise_std
        
        result_image =  tensor + noise

        save_tensor_image(result_image, filename = "/h/mikl123/Mitacs-aya/images_noised/asd.png")
        return result_image

    def generate_smooth_cam(self, image_tensor, inputs, vision_tower = False):
        smooth = True if self.noise_std != 0 else False
        image_tensor = inputs["pixel_values"].clone()
        base_cam, _ = self.generate_cam(image_tensor, smooth = smooth, vision_tower = vision_tower)
        
        smooth_cam = np.zeros_like(base_cam, dtype=np.float32)
        for _ in tqdm(range(self.num_samples)):
            noisy_image = self.add_noise(image_tensor.clone(), self.noise_std)
            noisy_cam, _ = self.generate_cam(noisy_image, smooth = smooth, vision_tower = vision_tower)
            smooth_cam += noisy_cam

        smooth_cam /= float(self.num_samples)
        smooth_cam = np.maximum(smooth_cam, 0)
        smooth_cam /= np.max(smooth_cam)
        # smooth_cam = cv2.resize(smooth_cam, (image_tensor.size(2), image_tensor.size(1)))
        smooth_cam = np.uint8(255 * smooth_cam)
        smooth_cam = cv2.applyColorMap(smooth_cam, cv2.COLORMAP_JET)

        original_image = self.unprocess_image(image_tensor.squeeze().cpu().numpy())
        superimposed_img = smooth_cam * 0.4 + original_image
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        return smooth_cam, superimposed_img
    
        
