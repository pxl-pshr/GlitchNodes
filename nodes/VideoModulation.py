# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import comfy.utils
import logging

logger = logging.getLogger(__name__)

class VideoModulation:
    """Applies CRT monitor and video modulation effects with RGB shifting."""
    CATEGORY = "GlitchNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "scan_density": ("INT", {"default": 4, "min": 2, "max": 10, "step": 1}),
                "rgb_shift": ("FLOAT", {"default": 0.015, "min": 0.0, "max": 0.05, "step": 0.001}),
                "brightness": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.0, "step": 0.1}),
                "dot_pattern": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_modulation"
    DESCRIPTION = "Simulates CRT monitor effects with scanlines, RGB shift, and dot patterns"

    def create_crt_pattern(self, height, width, density, device):
        # Create base coordinates
        y = torch.arange(height, device=device).float()
        x = torch.arange(width, device=device).float()
        
        # Create scan line pattern
        scan_pattern = torch.sin(y * np.pi * density / 2) * 0.5 + 0.5
        scan_pattern = scan_pattern.unsqueeze(1).expand(-1, width)
        
        # Create dot pattern
        dot_y = torch.sin(y.view(-1, 1) * np.pi * density / 2)
        dot_x = torch.sin(x.view(1, -1) * np.pi * density / 2)
        dot_pattern = (dot_y * dot_x) * 0.5 + 0.5
        
        return scan_pattern * dot_pattern

    def apply_rgb_shift(self, image, shift_amount):
        # Split channels
        r = image[..., 0:1]
        g = image[..., 1:2]
        b = image[..., 2:3]
        
        # Calculate shift in pixels
        height, width = image.shape[1:3]
        shift_h = int(height * shift_amount)
        shift_w = int(width * shift_amount)
        
        # Shift red channel up and right
        r_shifted = torch.roll(r, shifts=(shift_h, shift_w), dims=(1, 2))
        
        # Shift blue channel down and left
        b_shifted = torch.roll(b, shifts=(-shift_h, -shift_w), dims=(1, 2))
        
        # Combine channels with enhanced intensity
        r_shifted = torch.clamp(r_shifted * 1.3, 0, 1)
        g = torch.clamp(g * 0.8, 0, 1)
        b_shifted = torch.clamp(b_shifted * 1.3, 0, 1)
        
        return torch.cat([r_shifted, g, b_shifted], dim=-1)

    def apply_modulation(self, images, scan_density, rgb_shift, brightness, dot_pattern):
        logger.info("Starting video modulation processing...")
        try:
            device = images.device
            batch_size, height, width, channels = images.shape

            # Create base CRT pattern
            crt_pattern = self.create_crt_pattern(height, width, scan_density, device)

            # Process each image in the batch with progress bar
            pbar = comfy.utils.ProgressBar(batch_size)
            modulated_images = []
            for idx in range(batch_size):
                image = images[idx]

                # Apply RGB shift
                image = self.apply_rgb_shift(image.unsqueeze(0), rgb_shift).squeeze(0)

                # Apply CRT pattern
                pattern_strength = torch.lerp(
                    torch.ones_like(crt_pattern),
                    crt_pattern,
                    dot_pattern
                )
                image = image * pattern_strength.unsqueeze(-1)

                # Enhance brightness and contrast
                image = torch.pow(image * brightness, 1.4)

                # Add subtle noise
                noise = torch.randn_like(image) * 0.03
                image = torch.clamp(image + noise, 0, 1)

                # Enhance color separation
                image = torch.where(
                    image > 0.7,
                    torch.clamp(image * 1.3, 0, 1),
                    image * 0.7
                )

                modulated_images.append(image)
                pbar.update(1)

            logger.info("Video modulation processing completed")
            return (torch.stack(modulated_images),)
        except Exception as e:
            logger.error(f"Error in video modulation processing: {str(e)}")
            raise