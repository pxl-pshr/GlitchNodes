# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import torch
import numpy as np
from tqdm.auto import tqdm

class OrderedDithering:
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
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_dithering"
    CATEGORY = "image/effects"

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
        
        # Generate 16 artistic patterns for different brightness levels
        self.artistic_patterns = self._generate_artistic_patterns()

    def _generate_artistic_patterns(self):
        patterns = []
        base_size = 4  # 4x4 patterns
        
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
                    mask = np.random.rand(*variation.shape) < (i * 0.25)
                    variation[mask] = 1
                patterns.append(variation)
        
        # Normalize patterns
        return [p / p.max() for p in patterns]

    def convert_to_grayscale(self, image):
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])[..., np.newaxis]

    def get_artistic_pattern(self, brightness_level, pattern_contrast):
        # Map brightness to pattern index
        pattern_idx = int(brightness_level * (len(self.artistic_patterns) - 1))
        pattern = self.artistic_patterns[pattern_idx]
        
        # Apply contrast adjustment
        pattern = np.clip(pattern * pattern_contrast, 0, 1)
        return pattern

    def process_single_frame(self, image, pattern_type, num_colors, color_mode, 
                           pattern_contrast=1.0, threshold_offset=0.0, scale=1, invert=False):
        height, width = image.shape[:2]
        
        # Invert colors if requested
        if invert:
            image = 1.0 - image
        
        if color_mode == "Grayscale":
            image = self.convert_to_grayscale(image)
            
        # Get base pattern
        if pattern_type in self.bayer_matrices:
            matrix = self.bayer_matrices[pattern_type]
            matrix_size = matrix.shape[0]
            
            # Scale up the base pattern
            scaled_pattern = np.repeat(np.repeat(matrix, scale, axis=0), scale, axis=1)
            scaled_size = matrix_size * scale
            
            # Create full-size pattern
            pattern = np.tile(scaled_pattern, 
                            ((height + scaled_size - 1) // scaled_size,
                             (width + scaled_size - 1) // scaled_size))[:height, :width]
                             
            # Add threshold offset for animation
            if threshold_offset != 0:
                y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
                offset_matrix = (threshold_offset + (y_coords + x_coords) / (height + width)) % 1.0
                pattern = (pattern + offset_matrix) % 1.0
            
            # Handle color channels
            if len(image.shape) > 2:
                pattern = pattern[..., np.newaxis]
            
            # Apply dithering
            levels = np.linspace(0, 1, num_colors)
            dithered = image + (pattern * (1.0 / num_colors))
            quantized = np.digitize(dithered, bins=levels) - 1
            result = levels[quantized]
            
        else:  # Artistic mode
            matrix = self.bayer_matrices["4x4"]
            pattern = np.tile(matrix, 
                            ((height + 3) // 4,
                             (width + 3) // 4))[:height, :width]
            
            # Add animation offset if needed
            if threshold_offset != 0:
                y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
                offset_matrix = (threshold_offset + (y_coords + x_coords) / (height + width)) % 1.0
                pattern = (pattern + offset_matrix) % 1.0
                
            # Invert pattern if requested
            if invert:
                pattern = 1.0 - pattern
            
            # Handle color channels
            if len(image.shape) > 2:
                pattern = pattern[..., np.newaxis]
            
            # Apply artistic dithering
            levels = np.linspace(0, 1, num_colors)
            dithered = image + (pattern * pattern_contrast * (1.0 / num_colors))
            quantized = np.digitize(dithered, bins=levels) - 1
            result = levels[quantized]
        
        if color_mode == "Grayscale" and len(result.shape) == 2:
            result = np.repeat(result[..., np.newaxis], 3, axis=2)
            
        return result

    def apply_dithering(self, images, dither_type, color_mode, num_colors, pattern_size,
                       scale, pattern_contrast, frames, speed, wave_speed, invert):
        device = images.device
        images_np = images.cpu().numpy()
        
        pattern_type = pattern_size if dither_type != "Artistic" else "artistic"
        
        print(f"\n{'='*50}")
        print(f"Starting Ordered Dithering:")
        print(f"Mode: {dither_type}")
        print(f"Color Mode: {color_mode}")
        print(f"Pattern: {pattern_type}")
        print(f"Colors: {num_colors}")
        print(f"{'='*50}\n")
        
        try:
            if dither_type == "Standard" or dither_type == "Artistic":
                result = np.zeros_like(images_np)
                for b in range(len(images_np)):
                    result[b] = self.process_single_frame(
                        images_np[b], pattern_type, num_colors, color_mode,
                        pattern_contrast, scale=scale, invert=invert
                    )
                output = torch.from_numpy(result).to(device)
                
            else:  # Animated dithering
                output = torch.zeros((frames,) + images_np.shape[1:], device=device)
                
                with torch.no_grad():
                    for frame in tqdm(range(frames), desc="Generating dithered frames", leave=True):
                        # Direct copy of AnimatedSuperModulation's frame calculation
                        # They don't use frame/frames-1, just frame/frames and let it wrap naturally
                        cycle_progress = (frame / frames)
                        cycle_phase = cycle_progress * 2 * np.pi
                        wave_offset = -(cycle_phase * wave_speed)
                        
                        # Normalize to 0-1 range for our dithering
                        pattern_offset = ((wave_offset / (2 * np.pi)) + speed) % 1.0
                        
                        frame_result = np.zeros_like(images_np)
                        for b in range(len(images_np)):
                            frame_result[b] = self.process_single_frame(
                                images_np[b], pattern_type, num_colors, color_mode,
                                pattern_contrast, threshold_offset=pattern_offset, scale=scale, invert=invert
                            )
                        output[frame] = torch.from_numpy(frame_result[0]).to(device)
            
            print(f"\n{'='*50}")
            print(f"Dithering complete!")
            print(f"Output shape: {output.shape}")
            print(f"{'='*50}")
            
            return (output,)
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            raise e

NODE_CLASS_MAPPINGS = {
    "OrderedDithering": OrderedDithering
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OrderedDithering": "Ordered Dithering"
}