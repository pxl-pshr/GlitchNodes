# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import logging
import numpy as np
import torch
import comfy.utils

logger = logging.getLogger(__name__)

class VaporWave:
    """Apply vaporwave aesthetic color quantization."""
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold_dark": ("FLOAT", {
                    "default": 15/255,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "threshold_light": ("FLOAT", {
                    "default": 235/255,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "mid_threshold_1": ("FLOAT", {
                    "default": 60/255,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "mid_threshold_2": ("FLOAT", {
                    "default": 120/255,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "mid_threshold_3": ("FLOAT", {
                    "default": 180/255,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                # Color 1 (Cyan) - RGB components
                "color1_r": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color1_g": ("FLOAT", {"default": 0.722, "min": 0.0, "max": 1.0, "step": 0.01}),  # 184/255
                "color1_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Color 2 (Magenta) - RGB components
                "color2_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color2_g": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color2_b": ("FLOAT", {"default": 0.757, "min": 0.0, "max": 1.0, "step": 0.01}),  # 193/255
                # Color 3 (Purple) - RGB components
                "color3_r": ("FLOAT", {"default": 0.588, "min": 0.0, "max": 1.0, "step": 0.01}),  # 150/255
                "color3_g": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color3_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Color 4 (Aqua) - RGB components
                "color4_r": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color4_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color4_b": ("FLOAT", {"default": 0.976, "min": 0.0, "max": 1.0, "step": 0.01}),  # 249/255
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_vaporwave"
    CATEGORY = "GlitchNodes"
    DESCRIPTION = "Apply vaporwave aesthetic color quantization with customizable thresholds and color palette"

    def apply_vaporwave(self, image, threshold_dark, threshold_light,
                       mid_threshold_1, mid_threshold_2, mid_threshold_3,
                       color1_r, color1_g, color1_b,
                       color2_r, color2_g, color2_b,
                       color3_r, color3_g, color3_b,
                       color4_r, color4_g, color4_b):
        logger.info("Applying VaporWave effect...")

        # Convert to numpy array
        np_image = image.cpu().numpy().astype(np.float32)
        batch, height, width, channels = np_image.shape

        # Separate alpha and normalize to RGB
        alpha = None
        if channels == 4:
            alpha = np_image[..., 3:4]
            rgb = np_image[..., :3]
        elif channels == 1:
            rgb = np.repeat(np_image, 3, axis=-1)
        else:
            rgb = np_image[..., :3]

        # Create color arrays from individual components
        colors = np.array([
            [color1_r, color1_g, color1_b],
            [color2_r, color2_g, color2_b],
            [color3_r, color3_g, color3_b],
            [color4_r, color4_g, color4_b]
        ], dtype=np.float32)

        # Ensure thresholds are in ascending order so no band silently empties
        thresholds = sorted([threshold_dark, mid_threshold_1, mid_threshold_2,
                             mid_threshold_3, threshold_light])
        t_dark, mid_1, mid_2, mid_3, t_light = thresholds

        pbar = comfy.utils.ProgressBar(1)

        # Per-pixel luminance drives the palette mapping
        luminance = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[..., None]

        conditions = [
            (luminance <= t_dark),
            (luminance > t_dark) & (luminance <= mid_1),
            (luminance > mid_1) & (luminance <= mid_2),
            (luminance > mid_2) & (luminance <= mid_3),
            (luminance > mid_3) & (luminance <= t_light),
            (luminance > t_light)
        ]

        choices = [
            np.array([0, 0, 0], dtype=np.float32),
            colors[0],
            colors[1],
            colors[2],
            colors[3],
            np.array([1, 1, 1], dtype=np.float32)
        ]

        result = np.select(conditions, choices, rgb)
        pbar.update(1)

        # Reattach alpha channel if it exists
        if alpha is not None:
            result = np.concatenate([result, alpha], axis=-1)

        result = np.clip(result, 0.0, 1.0).astype(np.float32)

        logger.info("VaporWave effect completed!")

        # Convert back to torch tensor
        return (torch.from_numpy(result).to(image.device),)