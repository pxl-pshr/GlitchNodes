# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import numpy as np
import torch
from tqdm import tqdm


class VaporWave:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold_dark": (
                    "FLOAT",
                    {
                        "default": 15 / 255,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "threshold_light": (
                    "FLOAT",
                    {
                        "default": 235 / 255,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "mid_threshold_1": (
                    "FLOAT",
                    {
                        "default": 60 / 255,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "mid_threshold_2": (
                    "FLOAT",
                    {
                        "default": 120 / 255,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "mid_threshold_3": (
                    "FLOAT",
                    {
                        "default": 180 / 255,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_vaporwave"
    CATEGORY = "GlitchNodes"

    def apply_vaporwave(
        self,
        image,
        threshold_dark,
        threshold_light,
        mid_threshold_1,
        mid_threshold_2,
        mid_threshold_3,
        color1_r,
        color1_g,
        color1_b,
        color2_r,
        color2_g,
        color2_b,
        color3_r,
        color3_g,
        color3_b,
        color4_r,
        color4_g,
        color4_b,
    ):
        print("Applying VaporWave effect...")

        # Convert to numpy array
        np_image = image.cpu().numpy()
        batch, height, width, channels = np_image.shape

        # Create color arrays from individual components
        colors = np.array(
            [
                [color1_r, color1_g, color1_b],
                [color2_r, color2_g, color2_b],
                [color3_r, color3_g, color3_b],
                [color4_r, color4_g, color4_b],
            ]
        )

        # Reshape to 2D for processing
        flat_image = np_image.reshape(-1, channels)

        # Create result array
        result = np.zeros_like(flat_image)

        # Process in chunks with progress bar
        chunk_size = 100000  # Adjust based on memory constraints
        num_chunks = (flat_image.shape[0] + chunk_size - 1) // chunk_size

        with tqdm(total=num_chunks, desc="VAPORWAVING") as pbar:
            for i in range(0, flat_image.shape[0], chunk_size):
                chunk = flat_image[i : i + chunk_size]

                conditions = [
                    (chunk <= threshold_dark),
                    (chunk > threshold_dark) & (chunk <= mid_threshold_1),
                    (chunk > mid_threshold_1) & (chunk <= mid_threshold_2),
                    (chunk > mid_threshold_2) & (chunk <= mid_threshold_3),
                    (chunk > mid_threshold_3) & (chunk <= threshold_light),
                    (chunk > threshold_light),
                ]

                choices = [
                    [0, 0, 0],
                    colors[0],
                    colors[1],
                    colors[2],
                    colors[3],
                    [1, 1, 1],
                ]

                result[i : i + chunk_size] = np.select(conditions, choices, chunk)
                pbar.update(1)

        # Preserve alpha channel if it exists
        if channels == 4:
            result[:, 3] = flat_image[:, 3]

        # Reshape back to original dimensions
        result = result.reshape(batch, height, width, channels)

        print("VaporWave effect completed!")

        # Convert back to torch tensor
        return (torch.from_numpy(result).to(image.device),)


NODE_CLASS_MAPPINGS = {
    "VaporWave": VaporWave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VaporWave": "Vapor Wave Style",
}
