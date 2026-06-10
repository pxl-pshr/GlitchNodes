# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import torch
import numpy as np
import comfy.utils
import logging

logger = logging.getLogger(__name__)

class OrderedDithering:
    """Applies ordered dithering with multiple pattern types and optional animation."""
    DITHER_TYPES = ["Standard", "Artistic", "Animated"]
    COLOR_MODES = ["Color", "Grayscale"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "dither_type": (cls.DITHER_TYPES,),
                "color_mode": (cls.COLOR_MODES,),
                "num_colors": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 256,
                    "step": 1
                }),
                "pattern_size": (["2x2", "4x4", "8x8"],),
                "scale": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1
                }),
                "pattern_contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "frames": ("INT", {
                    "default": 60,
                    "min": 1,
                    "max": 300,
                    "step": 1,
                }),
                "speed": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "wave_speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "invert": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_dithering"
    CATEGORY = "GlitchNodes"
    DESCRIPTION = "Applies ordered dithering patterns with support for standard, artistic, and animated modes"

    def __init__(self):
        self.bayer_matrices = {
            "2x2": np.array([[0, 2],
                            [3, 1]]) / 4.0,

            "4x4": np.array([[0, 8, 2, 10],
                            [12, 4, 14, 6],
                            [3, 11, 1, 9],
                            [15, 7, 13, 5]]) / 16.0,

            "8x8": np.array([[0, 32, 8, 40, 2, 34, 10, 42],
                            [48, 16, 56, 24, 50, 18, 58, 26],
                            [12, 44, 4, 36, 14, 46, 6, 38],
                            [60, 28, 52, 20, 62, 30, 54, 22],
                            [3, 35, 11, 43, 1, 33, 9, 41],
                            [51, 19, 59, 27, 49, 17, 57, 25],
                            [15, 47, 7, 39, 13, 45, 5, 37],
                            [63, 31, 55, 23, 61, 29, 53, 21]]) / 64.0
        }

    def _generate_artistic_patterns(self, rng):
        patterns = []

        # Pattern 1: Dots
        p1 = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ])

        # Pattern 2: Lines horizontal
        p2 = np.array([
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0]
        ])

        # Pattern 3: Lines vertical
        p3 = np.array([
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0]
        ])

        # Pattern 4: Diagonal
        p4 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Generate variations with different densities
        base_patterns = [p1, p2, p3, p4]
        for p in base_patterns:
            # Create variations with different densities
            for i in range(4):
                variation = p.copy()
                if i > 0:
                    # Add more dots for higher brightness
                    mask = rng.random(variation.shape) < (i * 0.25)
                    variation[mask] = 1
                patterns.append(variation)

        # Normalize patterns (guard against all-zero patterns)
        return [p / p.max() if p.max() > 0 else p for p in patterns]

    def convert_to_grayscale(self, image):
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])[..., np.newaxis]

    def _build_bayer_pattern(self, pattern_size, scale, height, width):
        matrix = self.bayer_matrices.get(pattern_size, self.bayer_matrices["4x4"])
        scaled = np.repeat(np.repeat(matrix, scale, axis=0), scale, axis=1)
        sh, sw = scaled.shape
        return np.tile(scaled, ((height + sh - 1) // sh,
                                (width + sw - 1) // sw))[:height, :width]

    def _build_artistic_pattern(self, image, scale, pattern_contrast, patterns):
        # Select a pattern per block based on local brightness
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        height, width = gray.shape
        cell = 4 * scale
        blocks_y = (height + cell - 1) // cell
        blocks_x = (width + cell - 1) // cell

        padded = np.pad(gray, ((0, blocks_y * cell - height), (0, blocks_x * cell - width)), mode='edge')
        means = padded.reshape(blocks_y, cell, blocks_x, cell).mean(axis=(1, 3))

        idx = np.clip((means * (len(patterns) - 1)).astype(int), 0, len(patterns) - 1)
        stacked = np.stack(patterns).astype(np.float64)
        stacked = np.repeat(np.repeat(stacked, scale, axis=1), scale, axis=2)

        full = stacked[idx]  # (blocks_y, blocks_x, cell, cell)
        full = full.transpose(0, 2, 1, 3).reshape(blocks_y * cell, blocks_x * cell)[:height, :width]
        return np.clip(full * pattern_contrast, 0.0, 1.0)

    def process_single_frame(self, image, pattern, num_colors, color_mode, invert=False):
        # Invert colors if requested (image only, identical in all modes)
        if invert:
            image = 1.0 - image

        if color_mode == "Grayscale":
            image = self.convert_to_grayscale(image)

        # Centered dither scaled to level spacing, round to nearest level
        steps = max(1, num_colors - 1)
        dithered = image + (pattern[..., np.newaxis] - 0.5) / steps
        result = np.round(np.clip(dithered, 0.0, 1.0) * steps) / steps

        if color_mode == "Grayscale":
            result = np.repeat(result, 3, axis=2)

        return result

    def apply_dithering(self, images, dither_type, color_mode, num_colors, pattern_size,
                       scale, pattern_contrast, frames, speed, wave_speed, invert, seed=0):
        device = images.device
        images_np = images.cpu().numpy().astype(np.float64)
        B, H, W, C = images_np.shape

        # Split channels: process RGB, pass alpha through unchanged
        alpha = images_np[..., 3:4] if C == 4 else None
        if C == 1:
            rgb = np.repeat(images_np, 3, axis=-1)
        else:
            rgb = images_np[..., :3]

        logger.debug(f"Starting Ordered Dithering: mode={dither_type}, color_mode={color_mode}, "
                     f"pattern={pattern_size}, colors={num_colors}")

        rng = np.random.default_rng(seed)
        artistic_patterns = self._generate_artistic_patterns(rng) if dither_type == "Artistic" else None

        # Hoist the static parts of the pattern out of the frame loop
        base_pattern = None
        if dither_type != "Artistic":
            base_pattern = self._build_bayer_pattern(pattern_size, scale, H, W)
        offset_grid = None

        def frame_pattern(frame, threshold_offset=0.0):
            nonlocal offset_grid
            if dither_type == "Artistic":
                src = 1.0 - frame if invert else frame
                pattern = self._build_artistic_pattern(src, scale, pattern_contrast, artistic_patterns)
            else:
                pattern = base_pattern
            if threshold_offset != 0:
                if offset_grid is None:
                    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                    offset_grid = (y_coords + x_coords) / (H + W)
                pattern = (pattern + (threshold_offset + offset_grid) % 1.0) % 1.0
            return pattern

        try:
            if dither_type == "Standard" or dither_type == "Artistic":
                result = np.zeros_like(rgb)
                pbar = comfy.utils.ProgressBar(B)
                for b in range(B):
                    result[b] = self.process_single_frame(
                        rgb[b], frame_pattern(rgb[b]), num_colors, color_mode, invert=invert
                    )
                    pbar.update(1)
                if alpha is not None:
                    result = np.concatenate([result, alpha], axis=-1)

            else:  # Animated dithering
                if B > 1:
                    # Animate across the input batch: cycle the wave phase over frames
                    total = B
                    sources = rgb
                    out_alpha = alpha
                else:
                    # Single image: generate the requested number of frames from it
                    total = frames
                    sources = np.broadcast_to(rgb[0], (frames,) + rgb[0].shape)
                    out_alpha = np.repeat(alpha, frames, axis=0) if alpha is not None else None

                result = np.zeros((total, H, W, 3), dtype=np.float64)
                pbar = comfy.utils.ProgressBar(total)

                for frame in range(total):
                    cycle_progress = (frame / total)
                    cycle_phase = cycle_progress * 2 * np.pi
                    wave_offset = -(cycle_phase * wave_speed)

                    # Normalize to 0-1 range for our dithering
                    pattern_offset = ((wave_offset / (2 * np.pi)) + speed) % 1.0

                    result[frame] = self.process_single_frame(
                        sources[frame], frame_pattern(sources[frame], pattern_offset),
                        num_colors, color_mode, invert=invert
                    )
                    pbar.update(1)

                if out_alpha is not None:
                    result = np.concatenate([result, out_alpha], axis=-1)

            output = torch.from_numpy(result).float().clamp(0.0, 1.0).to(device)

            logger.debug(f"Dithering complete, output shape: {tuple(output.shape)}")

            return (output,)

        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            raise
