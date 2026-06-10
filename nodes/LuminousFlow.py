# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import torch
import numpy as np
import logging
from scipy.ndimage import gaussian_filter1d
import cv2
import comfy.utils

logger = logging.getLogger(__name__)

class LuminousFlow:
    """
    A ComfyUI node that transforms images into flowing luminous strands,
    creating an ethereal effect of light threads that follow the image's features.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "line_spacing": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 50,
                    "step": 1
                }),
                "line_thickness": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1
                }),
                "flow_intensity": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                }),
                "smoothing": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "glow_intensity": ("FLOAT", {
                    "default": 12.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1
                }),
                "darkness": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "vibrancy": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.1,
                    "max": 15.0,
                    "step": 0.1
                }),
                "glow_spread": ("INT", {
                    "default": 7,
                    "min": 0,
                    "max": 12,
                    "step": 1
                }),
                "contrast": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "create_luminous_flow"
    CATEGORY = "GlitchNodes"
    DESCRIPTION = "Transform images into flowing luminous strands with ethereal glow effects"

    def enhance_colors(self, colors, glow_intensity, vibrancy, contrast):
        """Enhanced color processing with neon effect, vectorized over a row of pixels"""
        # Convert to HSV for better color manipulation
        rgb_u8 = np.clip(colors * 255, 0, 255).astype(np.uint8).reshape(1, -1, 3)
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Super aggressive saturation enhancement
        hsv[..., 1] = np.clip(hsv[..., 1] * vibrancy * 1.5, 0, 255)

        # Enhanced value/brightness with extra pop
        hsv[..., 2] = np.clip(hsv[..., 2] * contrast * 1.5, 0, 255)

        # Convert back to RGB
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        enhanced = enhanced.reshape(-1, 3).astype(np.float32) / 255.0

        # Additional contrast enhancement with more aggressive gamma
        enhanced = np.power(enhanced, 0.7)
        enhanced = np.power(enhanced, 1 / contrast)

        # Extra boost to bright areas
        enhanced = np.where(enhanced > 0.5,
                            enhanced * 1.3,
                            enhanced * 0.7)
        enhanced = np.clip(enhanced, 0, 1)

        # Apply intensity scaling after hue/saturation handling so hue survives
        return enhanced * glow_intensity

    def process_image(self, img, params, batch_idx=0, total_batches=1):
        height, width = img.shape[:2]

        logger.info(f"Processing image {batch_idx + 1}/{total_batches}, size: {width}x{height}")

        rgb = np.ascontiguousarray(img[..., :3].astype(np.float32))

        # Enhanced preprocessing with more contrast
        intensity_map = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        intensity_map = cv2.GaussianBlur(intensity_map, (3, 3), 0)

        # More aggressive contrast in intensity map
        intensity_map = np.power(intensity_map, params["contrast"] * 1.2)
        intensity_map = (intensity_map - intensity_map.min()) / (intensity_map.max() - intensity_map.min() + 1e-7)

        # Create darker background
        canvas = rgb * params["darkness"] * 0.8

        max_lines = max(0, (height - 2 * params["line_spacing"]) // params["line_spacing"])
        if max_lines <= 0:
            logger.info("Image too small for line spacing, returning darkened background")
            return np.clip(canvas, 0, 1)

        line_positions = np.linspace(params["line_spacing"], height - params["line_spacing"], max_lines)

        logger.info(f"Generating {len(line_positions)} luminous lines")

        thickness = params["line_thickness"]
        xs = np.arange(width, dtype=np.float32)
        core = np.zeros((height, width, 3), dtype=np.float32)
        mask = np.zeros((height, width), dtype=np.float32)

        for pos in line_positions:
            y_pos = min(int(pos), height - 1)

            # Enhanced color processing for the whole row at once
            colors = self.enhance_colors(rgb[y_pos], params["glow_intensity"],
                                         params["vibrancy"], params["contrast"])

            displacement = -intensity_map[y_pos, :] * params["line_spacing"] * params["flow_intensity"]
            ys = np.clip(pos + displacement, 0, height - 1)

            if params["smoothing"] > 0:
                ys = gaussian_filter1d(ys, params["smoothing"])
                ys = np.clip(ys, 0, height - 1)

            pts = np.stack([xs, ys], axis=1).round().astype(np.int32).reshape(-1, 1, 2)

            # Rasterize the strand once, then colorize per column
            mask.fill(0)
            cv2.polylines(mask, [pts], False, 1.0, thickness, cv2.LINE_AA)

            y0 = max(0, int(ys.min()) - thickness - 2)
            y1 = min(height, int(ys.max()) + thickness + 3)
            core[y0:y1] += mask[y0:y1, :, None] * colors[None, :, :]

        # Accumulate strands and glow additively
        canvas = canvas + core * 1.5
        if params["glow_spread"] > 0:
            for i in range(1, params["glow_spread"] + 1):
                weight = (2.5 / (i + 0.2)) / params["glow_spread"]
                canvas += cv2.GaussianBlur(core, (0, 0), sigmaX=float(i)) * weight

        return np.clip(canvas, 0, 1)

    def create_luminous_flow(self, image, line_spacing, line_thickness, flow_intensity,
                           smoothing, glow_intensity, darkness, vibrancy, glow_spread,
                           contrast):
        batch_size = image.shape[0]

        logger.info(f"Starting Luminous Flow generation with batch size: {batch_size}, "
                   f"line_spacing={line_spacing}, flow_intensity={flow_intensity}, "
                   f"glow_intensity={glow_intensity}, vibrancy={vibrancy}, contrast={contrast}, "
                   f"glow_spread={glow_spread}")

        image_np = image.cpu().numpy().astype(np.float32)

        alpha = None
        if image_np.shape[-1] == 4:
            alpha = np.ascontiguousarray(image_np[..., 3:])
            image_np = image_np[..., :3]
        elif image_np.shape[-1] == 1:
            image_np = np.repeat(image_np, 3, axis=-1)

        params = {
            "line_spacing": line_spacing,
            "line_thickness": line_thickness,
            "flow_intensity": flow_intensity,
            "smoothing": smoothing,
            "glow_intensity": glow_intensity,
            "darkness": darkness,
            "vibrancy": vibrancy,
            "glow_spread": glow_spread,
            "contrast": contrast
        }

        output_batch = []
        pbar = comfy.utils.ProgressBar(batch_size)
        for i in range(batch_size):
            result = self.process_image(image_np[i], params, i, batch_size)
            result_tensor = torch.from_numpy(result).float()
            output_batch.append(result_tensor)
            pbar.update(1)

        logger.info("Luminous Flow processing complete")

        result = torch.stack(output_batch)
        if alpha is not None:
            result = torch.cat([result, torch.from_numpy(alpha)], dim=-1)
        return (result.float().clamp(0, 1).to(image.device),)
