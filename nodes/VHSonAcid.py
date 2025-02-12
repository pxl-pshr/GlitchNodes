# https://x.com/_pxlpshr

import torch
import numpy as np
from tqdm import tqdm

class VHSonAcid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "slice_size": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "offset_range": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                    "step": 1
                }),
                "color_shift": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "glitch_probability": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_glitch"
    CATEGORY = "image/processing"

    def create_slice_indices(self, height, slice_size):
        """Create random slice indices"""
        slices = []
        for i in range(0, height - slice_size, slice_size):
            if np.random.random() < self.current_params["glitch_probability"]:
                slices.append((i, i + slice_size))
        return slices

    def shift_slice(self, image, start, end, offset_range):
        """Shift a slice of the image horizontally"""
        offset = np.random.randint(-offset_range, offset_range)
        slice_data = image[start:end].copy()
        
        if offset > 0:
            image[start:end, offset:] = slice_data[:, :-offset]
            image[start:end, :offset] = slice_data[:, -offset:]
        elif offset < 0:
            image[start:end, :offset] = slice_data[:, -offset:]
            image[start:end, offset:] = slice_data[:, :-offset]
        
        return image

    def rgb_shift(self, image, amount):
        """Apply RGB channel shifting"""
        result = image.copy()
        channels = [0, 1, 2]  # RGB channels
        
        for i in channels:
            shift = int(np.random.uniform(-amount * 30, amount * 30))
            if shift > 0:
                result[..., i] = np.roll(image[..., i], shift, axis=1)
            elif shift < 0:
                result[..., i] = np.roll(image[..., i], shift, axis=1)
                
        return result

    def process_single_image(self, image):
        """Process a single image with glitch effects"""
        result = image.copy()
        height, width = image.shape[:2]
        
        # Create random slices
        slices = self.create_slice_indices(height, self.current_params["slice_size"])
        
        # Apply slice shifts
        for start, end in slices:
            result = self.shift_slice(result, start, end, self.current_params["offset_range"])
        
        # Apply color shifting
        if self.current_params["color_shift"] > 0:
            result = self.rgb_shift(result, self.current_params["color_shift"])
        
        return result

    def apply_glitch(self, images, slice_size, offset_range, color_shift, glitch_probability):
        device = images.device
        batch_size, height, width, channels = images.shape
        
        # Store parameters for use in other methods
        self.current_params = {
            "slice_size": slice_size,
            "offset_range": offset_range,
            "color_shift": color_shift,
            "glitch_probability": glitch_probability
        }
        
        output_batch = []
        
        print(f"\n{'='*50}")
        print(f"Applying glitch effects:")
        print(f"Batch size: {batch_size}")
        print(f"Parameters:")
        print(f"- Slice size: {slice_size}")
        print(f"- Offset range: {offset_range}")
        print(f"- Color shift: {color_shift}")
        print(f"- Glitch probability: {glitch_probability}")
        
        with tqdm(total=batch_size, desc="Processing images") as pbar:
            for b in range(batch_size):
                img = images[b].cpu().numpy()
                canvas = self.process_single_image(img)
                canvas_tensor = torch.from_numpy(canvas).float()
                output_batch.append(canvas_tensor)
                pbar.update(1)
        
        result = torch.stack(output_batch).to(device)
        
        print(f"\nProcessing complete!")
        print(f"{'='*50}")
        
        return (result,)