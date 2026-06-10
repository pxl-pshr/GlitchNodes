# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import logging
import torch
import numpy as np
import cv2
import comfy.utils

logger = logging.getLogger(__name__)

class Scanz:
    """Multi-effect glitch node with waves, scan lines, and color distortions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Base Effects
                "glitch_amount": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "channel_shift": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "pixel_sorting": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                # Wave Distortions
                "wave_amplitude": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "wave_frequency": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                }),
                "wave_speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                }),
                # Scan Lines
                "scan_lines": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "scan_drift": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "scan_curve": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                # Color Effects
                "color_drift": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "static_noise": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "edge_stretch": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "GlitchNodes"
    DESCRIPTION = "Apply multiple glitch effects including wave distortion, scan lines, and channel shifts"

    def apply_wave_distortion(self, images: np.ndarray, amplitude: float, frequency: float, speed: float) -> np.ndarray:
        batch, height, width = images.shape[:3]
        x_coords = np.arange(width)[np.newaxis, np.newaxis, :]
        y_coords = np.arange(height)[np.newaxis, :, np.newaxis]
        frame_idx = np.arange(batch)[:, np.newaxis, np.newaxis]
        wave = amplitude * 50 * np.sin(2 * np.pi * (
            frequency * y_coords / height +
            speed * x_coords / width +
            speed * frame_idx * 0.1
        ))

        src_x = np.clip(x_coords + wave.astype(np.intp), 0, width - 1).astype(np.intp)
        b_idx = np.arange(batch)[:, np.newaxis, np.newaxis]
        return images[b_idx, y_coords, src_x]

    def apply_scan_lines(self, images: np.ndarray, intensity: float, drift: float, curve: float, seed: int) -> np.ndarray:
        batch, height, width = images.shape[:3]
        result = images.copy()
        rgb = images[..., :3]

        # Calculate image features
        brightness = np.mean(rgb, axis=3)
        local_contrast = np.std(rgb, axis=3)

        # Create detail preservation mask
        preservation_mask = (local_contrast * 2.0 +
                           np.clip(1.0 - np.abs(brightness - 0.5) * 4, 0, 1))
        preservation_mask = np.clip(preservation_mask, 0.2, 1.0)

        # Create base coordinates
        y_coords = np.linspace(-1, 1, height, dtype=np.float32)
        x_coords = np.linspace(-1, 1, width, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Apply CRT-like curve distortion
        curve_factor = curve * 0.5
        dist = np.sqrt(xx**2 + yy**2)
        curved_xx = xx * (1 + curve_factor * dist**2)
        curved_yy = yy * (1 + curve_factor * dist**2)

        # Per-frame scan offset
        time_factor = np.array([
            np.random.default_rng(seed + i).random() * 10 for i in range(batch)
        ], dtype=np.float32)[:, np.newaxis, np.newaxis]

        # Create scan pattern with reduced intensity in extreme brightness areas
        scan_pattern = 0.5 + 0.3 * np.sin(2 * np.pi * (
            curved_yy[np.newaxis, ...] * 50 +
            drift * curved_xx[np.newaxis, ...] * 10 +
            brightness * 3 +
            time_factor
        ))

        # Calculate adaptive intensity
        scan_intensity = intensity * (0.7 + 0.3 * brightness)

        # Expand dimensions properly for broadcasting
        scan_pattern = scan_pattern[..., np.newaxis]
        scan_intensity = scan_intensity[..., np.newaxis]
        preservation_mask = preservation_mask[..., np.newaxis]

        # Apply the modulated scan pattern with detail preservation
        final_pattern = 1.0 - (scan_pattern * scan_intensity * (1.0 - preservation_mask))
        result[..., :3] = np.clip(rgb * final_pattern, 0, 1)

        return result

    def channel_shift_effect(self, images: np.ndarray, amount: float) -> np.ndarray:
        width = images.shape[2]
        result = images.copy()

        # Calculate local contrast to preserve features
        local_contrast = np.std(images[..., :3], axis=3)
        feature_mask = np.clip(local_contrast * 3.0, 0.2, 1.0)

        # Adjust shift amount based on local features
        max_shift = int(width * 0.1)
        base_shifts = [
            int(max_shift * amount * 0.5),  # Red
            -int(max_shift * amount * 0.3),  # Green
            int(max_shift * amount * 0.7)    # Blue
        ]

        pad_width = max(abs(shift) for shift in base_shifts)
        if pad_width == 0:
            return result
        padded = np.pad(images[..., :3], ((0, 0), (0, 0), (pad_width, pad_width), (0, 0)), mode='edge')

        for i in range(3):
            if base_shifts[i] != 0:
                shifted = np.roll(padded[..., i], base_shifts[i], axis=2)
                base_channel = shifted[:, :, pad_width:-pad_width]
                # Blend shifted and original based on feature mask
                result[..., i] = base_channel * feature_mask + images[..., i] * (1 - feature_mask)

        return np.clip(result, 0, 1)

    def apply_color_drift(self, images: np.ndarray, amount: float) -> np.ndarray:
        height, width = images.shape[1:3]
        result = images.copy()

        y_coords = np.linspace(0, height - 1, height)
        drift_pattern = (np.sin(y_coords * 0.1) * amount * 10).astype(np.intp)

        row_idx = np.arange(height)[:, np.newaxis]
        for c in range(3):
            shifts = drift_pattern * (c - 1)
            col_idx = (np.arange(width)[np.newaxis, :] - shifts[:, np.newaxis]) % width
            result[:, :, :, c] = images[:, row_idx, col_idx, c]

        return result

    def apply_edge_stretch(self, image: np.ndarray, amount: float) -> np.ndarray:
        rgb = image[..., :3]
        gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0

        height, width = image.shape[:2]
        displacement = cv2.GaussianBlur(edges, (21, 21), 5) * amount * 20

        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        src_x = np.clip(x_coords + displacement.astype(np.intp), 0, width - 1).astype(np.intp)
        return image[y_coords, src_x]

    def apply_static_noise(self, images: np.ndarray, amount: float, seed: int) -> np.ndarray:
        batch = images.shape[0]
        rgb = images[..., :3]
        noise = np.stack([
            np.random.default_rng(seed + i).normal(0, amount * 0.2, rgb.shape[1:]).astype(np.float32)
            for i in range(batch)
        ], axis=0)
        darkness = 1 - np.mean(rgb, axis=3, keepdims=True)
        weighted_noise = noise * darkness

        result = images.copy()
        result[..., :3] = np.clip(rgb + weighted_noise, 0, 1)
        return result

    def pixel_sort(self, images: np.ndarray, threshold: float) -> np.ndarray:
        batch, height, width = images.shape[:3]
        rgb = images[..., :3]

        # HSV value channel is the per-pixel max of RGB
        brightness = rgb.max(axis=3)
        mask = (brightness > threshold).reshape(batch * height, width)

        result = images.copy()
        res_flat = result.reshape(batch * height, width, -1)
        src_flat = images.reshape(batch * height, width, -1)

        # Sort masked pixels per row: argsort over a masked key array
        key = np.where(mask, np.mean(rgb, axis=3).reshape(batch * height, width), np.inf)
        order = np.argsort(key, axis=1, kind='stable')
        dest = np.argsort(~mask, axis=1, kind='stable')
        counts = mask.sum(axis=1)
        take = np.arange(width)[np.newaxis, :] < counts[:, np.newaxis]
        row_idx = np.repeat(np.arange(batch * height), counts)

        res_flat[row_idx, dest[take], :3] = src_flat[row_idx, order[take], :3]
        return result

    def apply_compression_artifacts(self, image: np.ndarray, quality: int = 50) -> np.ndarray:
        rgb = image[..., :3]
        temp_img = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded = cv2.imencode('.jpg', temp_img, encode_param)
        if not success or encoded is None:
            return image
        decoded = cv2.imdecode(encoded, 1)
        if decoded is None:
            return image
        result = image.copy()
        result[..., :3] = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return result

    def process_image(self, image, glitch_amount, channel_shift, pixel_sorting,
                     wave_amplitude, wave_frequency, wave_speed,
                     scan_lines, scan_drift, scan_curve,
                     color_drift, static_noise, edge_stretch, seed):
        if isinstance(image, torch.Tensor):
            batch_np = image.cpu().numpy().astype(np.float32)
        else:
            batch_np = np.asarray(image).astype(np.float32)

        if batch_np.shape[-1] == 1:
            batch_np = np.repeat(batch_np, 3, axis=-1)

        result = batch_np.copy()
        batch = result.shape[0]

        stages = sum([
            wave_amplitude > 0, color_drift > 0, edge_stretch > 0,
            channel_shift > 0, pixel_sorting > 0, scan_lines > 0,
            static_noise > 0, glitch_amount > 0
        ])
        pbar = comfy.utils.ProgressBar(max(stages, 1))

        if wave_amplitude > 0:
            result = self.apply_wave_distortion(result, wave_amplitude, wave_frequency, wave_speed)
            pbar.update(1)

        if color_drift > 0:
            result = self.apply_color_drift(result, color_drift)
            pbar.update(1)

        if edge_stretch > 0:
            result = np.stack([self.apply_edge_stretch(result[i], edge_stretch) for i in range(batch)], axis=0)
            pbar.update(1)

        if channel_shift > 0:
            result = self.channel_shift_effect(result, channel_shift)
            pbar.update(1)

        if pixel_sorting > 0:
            result = self.pixel_sort(result, pixel_sorting)
            pbar.update(1)

        if scan_lines > 0:
            result = self.apply_scan_lines(result, scan_lines, scan_drift, scan_curve, seed)
            pbar.update(1)

        if static_noise > 0:
            result = self.apply_static_noise(result, static_noise, seed)
            pbar.update(1)

        if glitch_amount > 0:
            quality = int(100 - (glitch_amount * 60))
            result = np.stack([self.apply_compression_artifacts(result[i], quality) for i in range(batch)], axis=0)
            pbar.update(1)

        result_batch = np.clip(result, 0, 1).astype(np.float32)
        device = image.device if isinstance(image, torch.Tensor) else "cpu"
        return (torch.from_numpy(result_batch).to(device),)
