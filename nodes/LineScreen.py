# https://x.com/_pxlpshr

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import math
import torch
from tqdm import tqdm

CATEGORY = "image/ProcessingFX"

class LineScreen:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "line_spacing": ("INT", {
                    "default": 4, 
                    "min": 2, 
                    "max": 20,
                    "step": 1,
                    "display": "slider"
                }),
                "angle": ("FLOAT", {
                    "default": -45.0, 
                    "min": -90.0, 
                    "max": 90.0,
                    "step": 0.5,
                    "display": "slider"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.1, 
                    "max": 0.9,
                    "step": 0.05,
                    "display": "slider"
                }),
                "contrast_boost": ("FLOAT", {
                    "default": 1.2, 
                    "min": 1.0, 
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "label": "Invert Colors"
                }),
                "line_color_r": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "line_color_g": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "line_color_b": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "bg_color_r": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "bg_color_g": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "bg_color_b": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_line_screen"
    
    CATEGORY = CATEGORY
    
    def create_line_pattern(self, width, height, spacing, angle, invert, line_color, bg_color):
        diagonal = int(math.sqrt(width**2 + height**2)) * 2
        pattern = Image.new('RGB', (diagonal, diagonal), tuple([int(c * 255) for c in bg_color]))
        draw = ImageDraw.Draw(pattern)
        
        num_lines = diagonal * 2 // spacing
        line_color_rgb = tuple([int(c * 255) for c in line_color])
        
        for i in range(-num_lines, num_lines):
            x = i * spacing
            draw.line([(x, -diagonal), (x, diagonal * 2)], fill=line_color_rgb, width=1)
        
        pattern = pattern.rotate(angle, resample=Image.BILINEAR, expand=True)
        
        left = (pattern.width - width) // 2
        top = (pattern.height - height) // 2
        right = left + width
        bottom = top + height
        
        return pattern.crop((left, top, right, bottom))
    
    def process_single_image(self, image_np, W, H, line_spacing, angle, threshold, contrast_boost, invert, 
                           line_color_r, line_color_g, line_color_b,
                           bg_color_r, bg_color_g, bg_color_b):
        if len(image_np.shape) == 3:
            img = Image.fromarray((image_np * 255).astype(np.uint8), mode='RGB')
        else:
            img = Image.fromarray((image_np * 255).astype(np.uint8), mode='L')
        
        img_gray = ImageOps.grayscale(img)
        if invert:
            img_gray = ImageOps.invert(img_gray)
        img_contrast = ImageEnhance.Contrast(img_gray).enhance(contrast_boost)
        
        line_color = [line_color_r, line_color_g, line_color_b]
        bg_color = [bg_color_r, bg_color_g, bg_color_b]
        
        if invert:
            line_color, bg_color = bg_color, line_color
        
        result = np.zeros((H, W, 3), dtype=np.float32)
        pattern = self.create_line_pattern(W, H, line_spacing, angle, invert, line_color, bg_color)
        pattern_array = np.array(pattern) / 255.0
        mask = np.array(img_contrast) < (255 * threshold)
        
        for c in range(3):
            result[..., c] = np.where(mask, pattern_array[..., c], bg_color[c])
        
        return result
    
    def apply_line_screen(self, image, line_spacing, angle, threshold, contrast_boost, invert,
                         line_color_r, line_color_g, line_color_b,
                         bg_color_r, bg_color_g, bg_color_b):
        device = image.device
        B, H, W, C = image.shape
        
        result_list = []
        image_np_batch = image.cpu().numpy()
        
        for b in tqdm(range(B), desc="Creating line screen", unit="img"):
            single_result = self.process_single_image(
                image_np_batch[b],
                W, H,
                line_spacing,
                angle,
                threshold,
                contrast_boost,
                invert,
                line_color_r, line_color_g, line_color_b,
                bg_color_r, bg_color_g, bg_color_b
            )
            result_list.append(single_result)
        
        result_array = np.stack(result_list)
        result_tensor = torch.from_numpy(result_array).to(device=device, dtype=torch.float32)
        
        return (result_tensor,)