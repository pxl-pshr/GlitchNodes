# Part 1: Imports and class definition
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from typing import Tuple, List, Optional

class Scanz:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_gpu = torch.cuda.is_available() and cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_gpu:
            print("GPU acceleration enabled")
        
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
                    "step": 0.05,
                    "group": "base_effects"
                }),
                "channel_shift": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "group": "base_effects"
                }),
                "pixel_sorting": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "group": "base_effects"
                }),
                # Wave Distortions
                "wave_amplitude": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "group": "wave_effects"
                }),
                "wave_frequency": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "group": "wave_effects"
                }),
                "wave_speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "group": "wave_effects"
                }),
                # Scan Lines
                "scan_lines": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "group": "scan_effects"
                }),
                "scan_drift": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "group": "scan_effects"
                }),
                "scan_curve": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "group": "scan_effects"
                }),
                # Color Effects
                "color_drift": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "group": "color_effects"
                }),
                "static_noise": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "group": "color_effects"
                }),
                "edge_stretch": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "group": "color_effects"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "image/effects"

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0
        return image

    def apply_wave_distortion(self, image: np.ndarray, amplitude: float, frequency: float, speed: float) -> np.ndarray:
        if amplitude == 0:
            return image
            
        height, width = image.shape[:2]
        result = np.zeros_like(image)
        
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        wave = amplitude * 50 * np.sin(2 * np.pi * (frequency * y_coords / height + speed * x_coords / width))
        
        for y in range(height):
            for x in range(width):
                offset = int(wave[y, x])
                new_x = min(max(x + offset, 0), width - 1)
                result[y, x] = image[y, new_x]
                
        return result

    def apply_scan_lines(self, image: np.ndarray, intensity: float, drift: float, curve: float) -> np.ndarray:
        if intensity == 0:
            return image
            
        height, width = image.shape[:2]
        result = image.copy()
        
        # Calculate image features
        brightness = np.mean(image, axis=2)
        local_contrast = np.std(image, axis=2)
        
        # Create detail preservation mask
        preservation_mask = (local_contrast * 2.0 + 
                           np.clip(1.0 - np.abs(brightness - 0.5) * 4, 0, 1))
        preservation_mask = np.clip(preservation_mask, 0.2, 1.0)
        
        # Create base coordinates
        y_coords = np.linspace(-1, 1, height)
        x_coords = np.linspace(-1, 1, width)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Apply CRT-like curve distortion
        curve_factor = curve * 0.5
        dist = np.sqrt(xx**2 + yy**2)
        curved_xx = xx * (1 + curve_factor * dist**2)
        curved_yy = yy * (1 + curve_factor * dist**2)
        
        # Create scan pattern with reduced intensity in extreme brightness areas
        time_factor = np.random.rand() * 10
        scan_pattern = 0.5 + 0.3 * np.sin(2 * np.pi * (
            curved_yy * 50 + 
            drift * curved_xx * 10 + 
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
        result = result * final_pattern
        
        return np.clip(result, 0, 1)

    def channel_shift_effect(self, image: np.ndarray, amount: float) -> np.ndarray:
        if amount == 0:
            return image
            
        height, width = image.shape[:2]
        result = np.zeros_like(image)
        
        # Calculate local contrast to preserve features
        local_contrast = np.std(image, axis=2)
        feature_mask = np.clip(local_contrast * 3.0, 0.2, 1.0)
        
        # Adjust shift amount based on local features
        max_shift = int(width * 0.1)
        base_shifts = [
            int(max_shift * amount * 0.5),  # Red
            -int(max_shift * amount * 0.3),  # Green
            int(max_shift * amount * 0.7)    # Blue
        ]
        
        pad_width = max(abs(shift) for shift in base_shifts)
        padded = np.pad(image, ((0, 0), (pad_width, pad_width), (0, 0)), mode='edge')
        
        for i in range(3):
            if base_shifts[i] != 0:
                shifted = np.roll(padded[..., i], base_shifts[i], axis=1)
                base_channel = shifted[:, pad_width:-pad_width]
                # Blend shifted and original based on feature mask
                result[..., i] = base_channel * feature_mask + image[..., i] * (1 - feature_mask)
            else:
                result[..., i] = image[..., i]
        
        # Enhance color contrast slightly
        result = np.clip(result * 1.1 - 0.05, 0, 1)
        
        return result

    def apply_color_drift(self, image: np.ndarray, amount: float) -> np.ndarray:
        if amount == 0:
            return image
            
        height, width = image.shape[:2]
        result = np.zeros_like(image)
        
        y_coords = np.linspace(0, height-1, height)
        drift_pattern = np.sin(y_coords * 0.1) * amount * 10
        
        for y in range(height):
            drift = int(drift_pattern[y])
            for c in range(3):
                shift = drift * (c - 1)
                rolled = np.roll(image[y, :, c], shift)
                result[y, :, c] = rolled
                
        return result
    def apply_edge_stretch(self, image: np.ndarray, amount: float) -> np.ndarray:
        if amount == 0:
            return image
            
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        
        height, width = image.shape[:2]
        displacement = cv2.GaussianBlur(edges, (21, 21), 5) * amount * 20
        
        result = np.zeros_like(image)
        for y in range(height):
            for x in range(width):
                offset = int(displacement[y, x])
                if offset > 0:
                    new_x = min(x + offset, width - 1)
                    result[y, x] = image[y, new_x]
                else:
                    result[y, x] = image[y, x]
                    
        return result

    def apply_static_noise(self, image: np.ndarray, amount: float) -> np.ndarray:
        if amount == 0:
            return image
            
        noise = np.random.normal(0, amount * 0.2, image.shape)
        darkness = 1 - np.mean(image, axis=2, keepdims=True)
        weighted_noise = noise * darkness
        
        result = image + weighted_noise
        return np.clip(result, 0, 1)

    def pixel_sort(self, image: np.ndarray, threshold: float) -> np.ndarray:
        if threshold == 0:
            return image
            
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        brightness = hsv[..., 2] / 255.0
        
        mask = brightness > threshold
        result = image.copy()
        
        for i in range(image.shape[0]):
            if mask[i].any():
                pixels = result[i, mask[i]]
                sorted_indices = np.argsort(np.mean(pixels, axis=1))
                result[i, mask[i]] = pixels[sorted_indices]
                
        return result

    def apply_compression_artifacts(self, image: np.ndarray, quality: int = 50) -> np.ndarray:
        temp_img = (image * 255).astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encoded = cv2.imencode('.jpg', temp_img, encode_param)
        decoded = cv2.imdecode(encoded, 1)
        return decoded.astype(np.float32) / 255.0

    def process_single_image(self, image: np.ndarray, params: dict) -> np.ndarray:
        image = self.normalize_image(image)
        result = image.copy()
        
        if params['wave_amplitude'] > 0:
            result = self.apply_wave_distortion(
                result,
                params['wave_amplitude'],
                params['wave_frequency'],
                params['wave_speed']
            )
            
        if params['color_drift'] > 0:
            result = self.apply_color_drift(result, params['color_drift'])
            
        if params['edge_stretch'] > 0:
            result = self.apply_edge_stretch(result, params['edge_stretch'])
            
        if params['channel_shift'] > 0:
            result = self.channel_shift_effect(result, params['channel_shift'])
            
        if params['pixel_sorting'] > 0:
            result = self.pixel_sort(result, params['pixel_sorting'])
            
        if params['scan_lines'] > 0:
            result = self.apply_scan_lines(
                result,
                params['scan_lines'],
                params['scan_drift'],
                params['scan_curve']
            )
            
        if params['static_noise'] > 0:
            result = self.apply_static_noise(result, params['static_noise'])
            
        if params['glitch_amount'] > 0:
            quality = int(100 - (params['glitch_amount'] * 60))
            result = self.apply_compression_artifacts(result, quality)
        
        return result

    def process_image(self, image, glitch_amount, channel_shift, pixel_sorting,
                     wave_amplitude, wave_frequency, wave_speed,
                     scan_lines, scan_drift, scan_curve,
                     color_drift, static_noise, edge_stretch):
        params = {
            'glitch_amount': glitch_amount,
            'channel_shift': channel_shift,
            'pixel_sorting': pixel_sorting,
            'wave_amplitude': wave_amplitude,
            'wave_frequency': wave_frequency,
            'wave_speed': wave_speed,
            'scan_lines': scan_lines,
            'scan_drift': scan_drift,
            'scan_curve': scan_curve,
            'color_drift': color_drift,
            'static_noise': static_noise,
            'edge_stretch': edge_stretch
        }
        
        if isinstance(image, torch.Tensor):
            batch_np = image.cpu().numpy()
        else:
            batch_np = image
            
        processed_images = []
        for i in tqdm(range(len(batch_np)), desc="Processing frames"):
            result = self.process_single_image(batch_np[i], params)
            processed_images.append(result)
            
        result_batch = np.stack(processed_images, axis=0)
        return (torch.from_numpy(result_batch).to(self.device),)