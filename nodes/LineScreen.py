# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import numpy as np
import logging
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import math
import torch
import comfy.utils

logger = logging.getLogger(__name__)

class LineScreen:
    """Convert images to line screen patterns with customizable angle and spacing"""
        
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
    CATEGORY = "GlitchNodes"
    DESCRIPTION = "Convert images to line screen patterns with customizable angle and spacing"
    
    def create_line_pattern(self, width, height, spacing, angle, line_color, bg_color):
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
    
    def process_single_image(self, image_np, pattern_array, bg_color, threshold, contrast_boost, invert):
        if image_np.ndim == 2:
            image_np = image_np[..., np.newaxis]
        if image_np.shape[-1] == 1:
            rgb = np.repeat(image_np, 3, axis=-1)
        else:
            rgb = image_np[..., :3]

        img = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), mode='RGB')

        img_gray = ImageOps.grayscale(img)
        if invert:
            img_gray = ImageOps.invert(img_gray)
        img_contrast = ImageEnhance.Contrast(img_gray).enhance(contrast_boost)

        mask = np.array(img_contrast) < (255 * threshold)
        bg = np.array(bg_color, dtype=np.float32)

        return np.where(mask[..., np.newaxis], pattern_array, bg).astype(np.float32)

    def apply_line_screen(self, image, line_spacing, angle, threshold, contrast_boost, invert,
                         line_color_r, line_color_g, line_color_b,
                         bg_color_r, bg_color_g, bg_color_b):
        device = image.device
        B, H, W, C = image.shape

        image_np_batch = image.cpu().numpy()
        alpha = image_np_batch[..., 3:4] if C == 4 else None

        line_color = [line_color_r, line_color_g, line_color_b]
        bg_color = [bg_color_r, bg_color_g, bg_color_b]

        # Pattern only depends on geometry and colors, build it once per batch
        pattern = self.create_line_pattern(W, H, line_spacing, angle, line_color, bg_color)
        pattern_array = (np.array(pattern, dtype=np.float32) / 255.0)[..., :3]

        result_list = []
        pbar = comfy.utils.ProgressBar(B)

        for b in range(B):
            single_result = self.process_single_image(
                image_np_batch[b],
                pattern_array,
                bg_color,
                threshold,
                contrast_boost,
                invert
            )
            result_list.append(single_result)
            pbar.update(1)

        result_array = np.stack(result_list)
        if alpha is not None:
            result_array = np.concatenate([result_array, alpha], axis=-1)
        result_tensor = torch.from_numpy(result_array).to(dtype=torch.float32).clamp(0.0, 1.0).to(device)

        return (result_tensor,)