# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import torch
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
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
        if shift_amount > 0:
            shift_h = max(1, shift_h)
            shift_w = max(1, shift_w)
        
        # Shift red channel up and right
        r_shifted = torch.roll(r, shifts=(shift_h, shift_w), dims=(1, 2))
        
        # Shift blue channel down and left
        b_shifted = torch.roll(b, shifts=(-shift_h, -shift_w), dims=(1, 2))
        
        # Combine channels with enhanced intensity
        r_shifted = torch.clamp(r_shifted * 1.3, 0, 1)
        g = torch.clamp(g * 0.8, 0, 1)
        b_shifted = torch.clamp(b_shifted * 1.3, 0, 1)
        
        return torch.cat([r_shifted, g, b_shifted], dim=-1)

    def apply_modulation(self, images, scan_density, rgb_shift, brightness, dot_pattern, seed):
        logger.info("Starting video modulation processing...")
        try:
            device = images.device
            batch_size, height, width, channels = images.shape

            image = images.float()

            # Normalize channels: expand grayscale, split off alpha
            alpha = None
            if channels == 1:
                image = image.expand(-1, -1, -1, 3).contiguous()
            elif channels == 4:
                alpha = image[..., 3:4]
                image = image[..., :3]
            else:
                image = image[..., :3]

            pbar = comfy.utils.ProgressBar(5)

            # Create base CRT pattern
            crt_pattern = self.create_crt_pattern(height, width, scan_density, device)

            # Apply RGB shift across the whole batch
            image = self.apply_rgb_shift(image, rgb_shift)
            pbar.update(1)

            # Apply CRT pattern
            pattern_strength = torch.lerp(
                torch.ones_like(crt_pattern),
                crt_pattern,
                dot_pattern
            )
            image = image * pattern_strength.unsqueeze(-1)
            pbar.update(1)

            # Enhance brightness and contrast
            image = torch.pow(image * brightness, 1.4)
            pbar.update(1)

            # Add subtle noise, seeded per frame
            noise = torch.stack([
                torch.randn(image.shape[1:], generator=torch.Generator().manual_seed((seed + idx) % (2**64)))
                for idx in range(batch_size)
            ]).to(device) * 0.03
            image = torch.clamp(image + noise, 0, 1)
            pbar.update(1)

            # Enhance color separation
            image = torch.where(
                image > 0.7,
                torch.clamp(image * 1.3, 0, 1),
                image * 0.7
            )
            pbar.update(1)

            if alpha is not None:
                image = torch.cat([image, alpha], dim=-1)

            logger.info("Video modulation processing completed")
            return (image.float().clamp(0, 1).to(device),)
        except Exception as e:
            logger.error(f"Error in video modulation processing: {str(e)}")
            raise