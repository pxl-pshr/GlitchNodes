# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import numpy as np
import torch
import logging
from PIL import Image
import random
import comfy.utils

logger = logging.getLogger(__name__)

class DitherMe:
    """
    A ComfyUI node that implements various dithering algorithms
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "algorithm": ([
                    "floyd_steinberg",
                    "jarvis_judice_ninke", 
                    "stucki",
                    "atkinson",
                    "burkes",
                    "sierra",
                    "sierra_2row",
                    "sierra_lite",
                    "ordered_2x2",
                    "ordered_4x4",
                    "ordered_8x8",
                    "bayer_2x2",
                    "bayer_4x4",
                    "bayer_8x8",
                    "random",
                    "threshold",
                    "halftone_dots",
                    "halftone_lines",
                    "blue_noise",
                    "white_noise",
                    "diffusion_horizontal",
                    "diffusion_vertical",
                    "diffusion_diagonal",
                    "clustered_dot_4x4",
                    "clustered_dot_8x8",
                    "dispersed_dot_4x4",
                    "dispersed_dot_8x8",
                    "void_and_cluster",
                    "hilbert_curve",
                    "spiral",
                    "zigzag",
                    "checkerboard",
                    "modulation",
                    "wave_interference",
                    "contour_lines",
                    "line_modulation"
                ],),
                "color_mode": (["monochrome", "duotone", "tritone", "indexed"],),
                "effect_size": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sharpen": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temporal_coherence": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "shadow_color": ("STRING", {"default": "#000000"}),
                "midtone_color": ("STRING", {"default": "#808080"}),
                "highlight_color": ("STRING", {"default": "#FFFFFF"}),
                "shadow_brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "midtone_brightness": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.01}),
                "highlight_brightness": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "palette_colors": ("INT", {"default": 2, "min": 2, "max": 256, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "dither"
    CATEGORY = "GlitchNodes"
    DESCRIPTION = "Apply various dithering algorithms including error diffusion and ordered dither patterns"

    def __init__(self):
        # Error diffusion matrices for various algorithms
        self.error_matrices = {
            "floyd_steinberg": {
                "matrix": [[0, 0, 7/16], 
                          [3/16, 5/16, 1/16]],
                "offset": (0, 1)
            },
            "jarvis_judice_ninke": {
                "matrix": [[0, 0, 0, 7/48, 5/48],
                          [3/48, 5/48, 7/48, 5/48, 3/48],
                          [1/48, 3/48, 5/48, 3/48, 1/48]],
                "offset": (0, 2)
            },
            "stucki": {
                "matrix": [[0, 0, 0, 8/42, 4/42],
                          [2/42, 4/42, 8/42, 4/42, 2/42],
                          [1/42, 2/42, 4/42, 2/42, 1/42]],
                "offset": (0, 2)
            },
            "atkinson": {
                "matrix": [[0, 0, 1/8, 1/8],
                          [1/8, 1/8, 1/8, 0],
                          [0, 1/8, 0, 0]],
                "offset": (0, 1)
            },
            "burkes": {
                "matrix": [[0, 0, 0, 8/32, 4/32],
                          [2/32, 4/32, 8/32, 4/32, 2/32]],
                "offset": (0, 2)
            },
            "sierra": {
                "matrix": [[0, 0, 0, 5/32, 3/32],
                          [2/32, 4/32, 5/32, 4/32, 2/32],
                          [0, 2/32, 3/32, 2/32, 0]],
                "offset": (0, 2)
            },
            "sierra_2row": {
                "matrix": [[0, 0, 0, 4/16, 3/16],
                          [1/16, 2/16, 3/16, 2/16, 1/16]],
                "offset": (0, 2)
            },
            "sierra_lite": {
                "matrix": [[0, 0, 2/4],
                          [1/4, 1/4, 0]],
                "offset": (0, 1)
            }
        }
        
        # Ordered dither matrices
        self.ordered_matrices = {
            "2x2": np.array([[0, 2], [3, 1]]) / 4.0,
            "4x4": np.array([[0, 8, 2, 10],
                            [12, 4, 14, 6],
                            [3, 11, 1, 9],
                            [15, 7, 13, 5]]) / 16.0,
            "8x8": self._generate_bayer_matrix(8)
        }
    
    def _generate_bayer_matrix(self, n):
        """Generate Bayer matrix of size n x n"""
        if n == 2:
            return np.array([[0, 2], [3, 1]]) / 4.0
        else:
            smaller = self._generate_bayer_matrix(n // 2)
            return np.block([[4 * smaller, 4 * smaller + 2],
                           [4 * smaller + 3, 4 * smaller + 1]]) / (n * n)
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    
    def _apply_preprocessing(self, image, sharpen, blur, noise):
        """Apply preprocessing effects to image"""
        if blur > 0:
            from scipy.ndimage import gaussian_filter
            image = gaussian_filter(image, sigma=blur)
        
        if sharpen > 0:
            from scipy.ndimage import convolve
            kernel = np.array([[-1, -1, -1],
                              [-1, 9 + sharpen * 8, -1],
                              [-1, -1, -1]])
            kernel = kernel / kernel.sum()
            if len(image.shape) == 3:
                for i in range(image.shape[2]):
                    image[:, :, i] = convolve(image[:, :, i], kernel)
            else:
                image = convolve(image, kernel)
        
        if noise > 0:
            noise_array = np.random.normal(0, noise * 0.1, image.shape)
            image = np.clip(image + noise_array, 0, 1)
        
        return image
    
    def _error_diffusion_dither(self, image, algorithm, threshold=0.5):
        """Apply error diffusion dithering"""
        if algorithm not in self.error_matrices:
            return image

        matrix_info = self.error_matrices[algorithm]
        matrix = np.array(matrix_info["matrix"])
        offset = matrix_info["offset"]

        height, width = image.shape[:2]
        if len(image.shape) == 3:
            # Convert to grayscale for dithering
            gray = np.dot(image, [0.299, 0.587, 0.114])
        else:
            gray = image.copy()

        output = np.zeros_like(gray)

        for y in range(height):
            for x in range(width):
                old_pixel = gray[y, x]
                new_pixel = 1.0 if old_pixel > threshold else 0.0
                output[y, x] = new_pixel
                error = old_pixel - new_pixel

                # Distribute error to neighboring pixels
                for dy in range(len(matrix)):
                    for dx in range(len(matrix[0])):
                        if matrix[dy][dx] == 0:
                            continue

                        ny = y + dy
                        nx = x + dx - offset[1]

                        if 0 <= ny < height and 0 <= nx < width:
                            gray[ny, nx] += error * matrix[dy][dx]

        return output
    
    def _ordered_dither(self, image, matrix_type, threshold=0.5, effect_size=1.0):
        """Apply ordered dithering"""
        matrix_size = matrix_type.split('_')[-1]
        if matrix_size not in self.ordered_matrices:
            matrix_size = "4x4"

        matrix = self.ordered_matrices[matrix_size] * effect_size

        height, width = image.shape[:2]
        if len(image.shape) == 3:
            gray = np.dot(image, [0.299, 0.587, 0.114])
        else:
            gray = image.copy()

        # Tile the matrix to cover the full image
        mh, mw = matrix.shape
        tiled = np.tile(matrix, ((height + mh - 1) // mh, (width + mw - 1) // mw))[:height, :width]

        return (gray > tiled).astype(np.float64)
    
    def _apply_color_mapping(self, dithered, color_mode, shadow_color, midtone_color, 
                           highlight_color, shadow_brightness, midtone_brightness, 
                           highlight_brightness):
        """Apply color mapping based on mode"""
        height, width = dithered.shape
        
        if color_mode == "monochrome":
            # Simple black and white
            return np.stack([dithered, dithered, dithered], axis=-1)
        
        elif color_mode == "duotone":
            # Two colors
            shadow_rgb = np.array(self._hex_to_rgb(shadow_color))
            highlight_rgb = np.array(self._hex_to_rgb(highlight_color))
            
            # Adjust brightness
            shadow_rgb = np.clip(shadow_rgb + shadow_brightness, 0, 1)
            highlight_rgb = np.clip(highlight_rgb + highlight_brightness, 0, 1)
            
            output = np.zeros((height, width, 3))
            output[dithered == 0] = shadow_rgb
            output[dithered == 1] = highlight_rgb
            
            return output
        
        elif color_mode == "tritone":
            # Three colors with midtones
            shadow_rgb = np.array(self._hex_to_rgb(shadow_color))
            midtone_rgb = np.array(self._hex_to_rgb(midtone_color))
            highlight_rgb = np.array(self._hex_to_rgb(highlight_color))

            # Adjust brightness
            shadow_rgb = np.clip(shadow_rgb + shadow_brightness, 0, 1)
            midtone_rgb = np.clip(midtone_rgb + midtone_brightness, 0, 1)
            highlight_rgb = np.clip(highlight_rgb + highlight_brightness, 0, 1)

            # Dithered values are binary (0 or 1). For tritone, use a
            # checkerboard pattern to create midtone regions where highlight
            # pixels alternate with midtone color.
            output = np.zeros((height, width, 3))
            checker = ((np.arange(height)[:, None] + np.arange(width)[None, :]) % 2).astype(bool)

            output[dithered == 0] = shadow_rgb
            output[(dithered == 1) & checker] = highlight_rgb
            output[(dithered == 1) & ~checker] = midtone_rgb

            return output
        
        else:  # monochrome fallback
            return np.stack([dithered, dithered, dithered], axis=-1)
    
    def _special_dither(self, image, algorithm, threshold=0.5, effect_size=1.0):
        """Handle special dithering algorithms"""
        height, width = image.shape[:2]
        if len(image.shape) == 3:
            gray = np.dot(image, [0.299, 0.587, 0.114])
        else:
            gray = image.copy()
        
        output = np.zeros_like(gray)
        
        if algorithm == "random":
            # Random dithering (vectorized)
            random_thresholds = np.random.random((height, width)) * effect_size
            output = (gray > random_thresholds).astype(np.float64)
        
        elif algorithm == "threshold":
            # Simple threshold
            output = (gray > threshold).astype(float)
        
        elif algorithm == "halftone_dots":
            # Simulate halftone dots
            dot_size = int(4 * effect_size)
            for y in range(0, height, dot_size):
                for x in range(0, width, dot_size):
                    # Get average value in block
                    block = gray[y:y+dot_size, x:x+dot_size]
                    avg = np.mean(block)
                    
                    # Draw circle based on average
                    radius = int(dot_size * avg / 2)
                    cy, cx = y + dot_size // 2, x + dot_size // 2
                    
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            if dy*dy + dx*dx <= radius*radius:
                                ny, nx = cy + dy, cx + dx
                                if 0 <= ny < height and 0 <= nx < width:
                                    output[ny, nx] = 1.0
        
        elif algorithm == "modulation":
            # Advanced modulation-style dithering with smooth wave patterns (vectorized)
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
            base_freq = 0.15 / effect_size

            wave_freq = base_freq * (0.5 + gray * 1.5)
            modulation_strength = gray * 0.5
            x_offset = np.sin(yy * base_freq * 0.2) * modulation_strength * 50
            modulated_wave = np.sin(yy * wave_freq + xx * 0.01 + x_offset)

            vertical_influence = np.sin(xx * base_freq * 0.3) * 0.2
            combined = modulated_wave + vertical_influence

            wave_threshold = (combined > 0).astype(np.float64) * 0.4 + 0.3
            output = (gray > wave_threshold).astype(np.float64)
        
        elif algorithm == "wave_interference":
            # Create smooth wave interference patterns (vectorized)
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
            line_frequency = 0.1 / effect_size

            primary_freq = line_frequency * (1 + gray * 2)
            wave_amplitude = 20 * effect_size
            x_displacement = np.sin(yy * line_frequency * 0.5) * wave_amplitude * gray

            main_wave = np.sin(yy * primary_freq + x_displacement * 0.01)
            secondary_wave = np.sin(xx * line_frequency * 0.3 + yy * 0.01) * gray * 0.5

            combined = main_wave + secondary_wave
            wave_value = (combined + 1) * 0.5

            output = (gray > wave_value * 0.6 + 0.2).astype(np.float64)
        
        elif algorithm == "contour_lines":
            # Create contour-following line patterns (vectorized)
            from scipy import ndimage
            grad_x = ndimage.sobel(gray, axis=1)
            grad_y = ndimage.sobel(gray, axis=0)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

            yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
            freq = 0.05 * effect_size

            angle = np.arctan2(grad_y, grad_x)
            # Lines perpendicular to gradient
            contour_pattern = np.sin((xx * np.sin(angle) - yy * np.cos(angle)) * freq * (1 + gradient_mag * 20))
            # Horizontal lines for flat areas
            flat_pattern = np.sin(yy * freq * (1 + gray * 5))
            # Blend based on gradient strength
            has_gradient = gradient_mag > 0.01
            line_pattern = np.where(has_gradient, contour_pattern, flat_pattern)

            modulation = np.sin(xx * 0.01 * effect_size) * 0.2
            threshold_val = 0.5 + (line_pattern + modulation) * 0.3
            output = (gray > threshold_val).astype(np.float64)
        
        elif algorithm == "line_modulation":
            # Creates clean modulated line patterns like in the reference
            line_spacing = max(1, int(5 / effect_size))  # Controls line density
            
            # First pass: create base line pattern
            for y in range(height):
                if y % line_spacing == 0:  # Only process every nth line
                    # Initialize line with basic threshold
                    for x in range(width):
                        output[y, x] = 1.0 if gray[y, x] > threshold else 0.0
            
            # Second pass: modulate the lines based on image content
            for y in range(0, height, line_spacing):
                # Track the line position with modulation
                prev_offset = 0
                
                for x in range(width):
                    # Sample brightness around current position
                    sample_radius = 3
                    local_brightness = 0
                    sample_count = 0
                    
                    for dy in range(-sample_radius, sample_radius + 1):
                        for dx in range(-sample_radius, sample_radius + 1):
                            sy = y + dy
                            sx = x + dx
                            if 0 <= sy < height and 0 <= sx < width:
                                local_brightness += gray[sy, sx]
                                sample_count += 1
                    
                    if sample_count > 0:
                        local_brightness /= sample_count
                    
                    # Calculate line offset based on brightness
                    offset = int((local_brightness - 0.5) * line_spacing * 2)
                    offset = max(-line_spacing + 1, min(line_spacing - 1, offset))
                    
                    # Smooth the offset to prevent jagged lines
                    offset = int(prev_offset * 0.7 + offset * 0.3)
                    prev_offset = offset
                    
                    # Draw the modulated line
                    for line_y in range(y - 1, y + 2):  # 3-pixel wide line
                        target_y = line_y + offset
                        if 0 <= target_y < height:
                            # Anti-aliasing: fade at line edges
                            if line_y == y:
                                output[target_y, x] = 1.0 if gray[target_y, x] > threshold * 0.5 else 0.0
                            else:
                                output[target_y, x] = 1.0 if gray[target_y, x] > threshold * 0.7 else 0.0
        
        else:
            # Default to threshold for unimplemented algorithms
            output = (gray > threshold).astype(float)
        
        return output
    
    def dither(self, image, algorithm, color_mode, effect_size, threshold, 
               sharpen, blur, noise, temporal_coherence, shadow_color="#000000", 
               midtone_color="#808080", highlight_color="#FFFFFF", shadow_brightness=0.0, 
               midtone_brightness=0.5, highlight_brightness=1.0, palette_colors=2):
        """Main dithering function with batch processing support and progress bar"""
        
        # Convert from torch tensor to numpy array
        if isinstance(image, torch.Tensor):
            batch_np = image.cpu().numpy()
        else:
            batch_np = np.array(image)
        
        # Ensure we have batch dimension
        if len(batch_np.shape) == 3:
            batch_np = np.expand_dims(batch_np, 0)
        
        batch_size = batch_np.shape[0]
        results = []
        previous_dithered = None

        # Create progress bar
        pbar = comfy.utils.ProgressBar(batch_size)

        # Process each frame in the batch with progress bar
        for i in range(batch_size):
            frame = batch_np[i]
            
            # Ensure image is in [0, 1] range
            if frame.max() > 1.0:
                frame = frame / 255.0
            
            # Apply preprocessing
            frame = self._apply_preprocessing(frame, sharpen, blur, noise)
            
            # Apply dithering algorithm
            if algorithm in self.error_matrices:
                dithered = self._error_diffusion_dither(frame, algorithm, threshold)
            elif "ordered" in algorithm or "bayer" in algorithm:
                dithered = self._ordered_dither(frame, algorithm, threshold, effect_size)
            else:
                dithered = self._special_dither(frame, algorithm, threshold, effect_size)
            
            # Apply temporal coherence if enabled and not the first frame
            if temporal_coherence > 0 and previous_dithered is not None and i > 0:
                # Blend with previous frame to reduce flickering
                dithered = (1 - temporal_coherence) * dithered + temporal_coherence * previous_dithered
            
            previous_dithered = dithered.copy()
            
            # Apply color mapping
            result = self._apply_color_mapping(
                dithered, color_mode, shadow_color, midtone_color, highlight_color,
                shadow_brightness, midtone_brightness, highlight_brightness
            )

            results.append(result)
            pbar.update(1)

        # Stack all results into a batch
        batch_result = np.stack(results, axis=0)
        
        # Convert back to torch tensor
        result_tensor = torch.from_numpy(batch_result).float()
        
        return (result_tensor,)