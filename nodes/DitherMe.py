# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import numpy as np
import torch
import logging
from PIL import Image
import comfy.utils

logger = logging.getLogger(__name__)

class DitherMe:
    """
    A ComfyUI node that implements various dithering algorithms
    """

    _VOID_CLUSTER_CACHE = {}

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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
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
            },
            "diffusion_horizontal": {
                "matrix": [[0, 0, 1.0]],
                "offset": (0, 1)
            },
            "diffusion_vertical": {
                "matrix": [[0],
                          [1.0]],
                "offset": (0, 0)
            },
            "diffusion_diagonal": {
                "matrix": [[0, 0],
                          [0, 1.0]],
                "offset": (0, 0)
            }
        }

        # Ordered dither matrices
        self.ordered_matrices = {
            "2x2": self._generate_bayer_matrix(2),
            "4x4": self._generate_bayer_matrix(4),
            "8x8": self._generate_bayer_matrix(8)
        }

        # Classic halftone threshold matrices
        self.halftone_matrices = {
            "clustered_dot_4x4": np.array([[12, 5, 6, 13],
                                           [4, 0, 1, 7],
                                           [11, 3, 2, 8],
                                           [15, 10, 9, 14]]) / 16.0,
            "clustered_dot_8x8": np.array([[24, 10, 12, 26, 35, 47, 49, 37],
                                           [8, 0, 2, 14, 45, 59, 61, 51],
                                           [22, 6, 4, 16, 43, 57, 63, 53],
                                           [30, 20, 18, 28, 33, 41, 55, 39],
                                           [34, 46, 48, 36, 25, 11, 13, 27],
                                           [44, 58, 60, 50, 9, 1, 3, 15],
                                           [42, 56, 62, 52, 23, 7, 5, 17],
                                           [32, 40, 54, 38, 31, 21, 19, 29]]) / 64.0,
            "dispersed_dot_4x4": self._generate_bayer_matrix(4),
            "dispersed_dot_8x8": self._generate_bayer_matrix(8),
        }

    def _generate_bayer_integers(self, n):
        """Generate integer Bayer matrix of size n x n with values 0..n*n-1"""
        if n == 2:
            return np.array([[0, 2], [3, 1]])
        smaller = self._generate_bayer_integers(n // 2)
        return np.block([[4 * smaller, 4 * smaller + 2],
                         [4 * smaller + 3, 4 * smaller + 1]])

    def _generate_bayer_matrix(self, n):
        """Generate normalized Bayer matrix of size n x n"""
        matrix = self._generate_bayer_integers(n)
        return matrix / matrix.size

    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    def _to_gray(self, image):
        """Convert an image array to grayscale"""
        if image.ndim == 3:
            return np.dot(image[..., :3], [0.299, 0.587, 0.114])
        return image.astype(np.float64, copy=True)

    def _apply_preprocessing(self, image, sharpen, blur, noise, rng):
        """Apply preprocessing effects to image"""
        if blur > 0:
            from scipy.ndimage import gaussian_filter
            if image.ndim == 3:
                image = gaussian_filter(image, sigma=(blur, blur, 0))
            else:
                image = gaussian_filter(image, sigma=blur)

        if sharpen > 0:
            from scipy.ndimage import convolve
            kernel = np.array([[0, -sharpen, 0],
                              [-sharpen, 1 + 4 * sharpen, -sharpen],
                              [0, -sharpen, 0]])
            if image.ndim == 3:
                image = np.stack([convolve(image[:, :, i], kernel)
                                  for i in range(image.shape[2])], axis=-1)
            else:
                image = convolve(image, kernel)
            image = np.clip(image, 0, 1)

        if noise > 0:
            noise_array = rng.normal(0, noise * 0.1, image.shape)
            image = np.clip(image + noise_array, 0, 1)

        return image

    def _error_diffusion_dither(self, image, algorithm, threshold=0.5):
        """Apply error diffusion dithering"""
        if algorithm not in self.error_matrices:
            return self._to_gray(image)

        matrix_info = self.error_matrices[algorithm]
        taps = matrix_info.get("taps")
        if taps is None:
            offset = matrix_info["offset"]
            taps = [(dy, dx - offset[1], float(w))
                    for dy, row in enumerate(matrix_info["matrix"])
                    for dx, w in enumerate(row) if w != 0]
            matrix_info["taps"] = taps

        gray = self._to_gray(image)
        height, width = gray.shape
        output = np.zeros_like(gray)

        for y in range(height):
            row = gray[y]
            out_row = output[y]
            for x in range(width):
                old_pixel = row[x]
                new_pixel = 1.0 if old_pixel > threshold else 0.0
                out_row[x] = new_pixel
                error = old_pixel - new_pixel

                for dy, dx, weight in taps:
                    ny = y + dy
                    nx = x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        gray[ny, nx] += error * weight

        return output

    def _matrix_dither(self, gray, matrix, effect_size=1.0):
        """Threshold against a tiled matrix; effect_size scales the pattern size"""
        scale = max(1, int(round(effect_size)))
        if scale > 1:
            matrix = np.repeat(np.repeat(matrix, scale, axis=0), scale, axis=1)

        height, width = gray.shape
        mh, mw = matrix.shape
        tiled = np.tile(matrix, ((height + mh - 1) // mh, (width + mw - 1) // mw))[:height, :width]

        return (gray > tiled).astype(np.float64)

    def _ordered_dither(self, image, matrix_type, threshold=0.5, effect_size=1.0):
        """Apply ordered dithering"""
        matrix_size = matrix_type.split('_')[-1]
        if matrix_size not in self.ordered_matrices:
            matrix_size = "4x4"

        return self._matrix_dither(self._to_gray(image), self.ordered_matrices[matrix_size], effect_size)

    def _get_void_cluster_matrix(self, size):
        """Generate (and cache) a void-and-cluster threshold matrix"""
        cached = DitherMe._VOID_CLUSTER_CACHE.get(size)
        if cached is not None:
            return cached

        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(0)
        n = size * size
        ones = max(1, n // 10)

        flat = np.zeros(n, dtype=bool)
        flat[:ones] = True
        rng.shuffle(flat)
        pattern = flat.reshape(size, size)

        def blurred(p):
            return gaussian_filter(p.astype(np.float64), sigma=1.5, mode='wrap')

        # Phase 0: relax initial pattern (swap tightest cluster with largest void)
        for _ in range(n):
            b = blurred(pattern)
            cluster = np.unravel_index(np.argmax(np.where(pattern, b, -np.inf)), pattern.shape)
            pattern[cluster] = False
            b = blurred(pattern)
            void = np.unravel_index(np.argmin(np.where(pattern, np.inf, b)), pattern.shape)
            if void == cluster:
                pattern[cluster] = True
                break
            pattern[void] = True

        rank = np.zeros((size, size), dtype=np.int64)

        # Phase 1: rank initial minority pixels by removing tightest clusters
        work = pattern.copy()
        for r in range(ones - 1, -1, -1):
            b = blurred(work)
            cluster = np.unravel_index(np.argmax(np.where(work, b, -np.inf)), work.shape)
            work[cluster] = False
            rank[cluster] = r

        # Phase 2: rank remaining pixels by filling largest voids
        work = pattern.copy()
        for r in range(ones, n):
            b = blurred(work)
            void = np.unravel_index(np.argmin(np.where(work, np.inf, b)), work.shape)
            work[void] = True
            rank[void] = r

        matrix = (rank + 0.5) / n
        DitherMe._VOID_CLUSTER_CACHE[size] = matrix
        return matrix

    def _hilbert_coordinates(self, n):
        """Vectorized d -> (x, y) conversion for a Hilbert curve over an n x n grid"""
        d = np.arange(n * n, dtype=np.int64)
        t = d.copy()
        x = np.zeros_like(d)
        y = np.zeros_like(d)
        s = 1
        while s < n:
            rx = (t >> 1) & 1
            ry = (t ^ rx) & 1
            swap = ry == 0
            flip = swap & (rx == 1)
            xf = np.where(flip, s - 1 - x, x)
            yf = np.where(flip, s - 1 - y, y)
            x, y = np.where(swap, yf, xf), np.where(swap, xf, yf)
            x = x + s * rx
            y = y + s * ry
            t >>= 2
            s <<= 1
        return x, y

    def _hilbert_dither(self, image, threshold=0.5):
        """Error diffusion along Hilbert curve scan order"""
        gray = self._to_gray(image)
        height, width = gray.shape
        n = 1 << max(0, (max(height, width) - 1).bit_length())
        xs, ys = self._hilbert_coordinates(n)
        valid = (xs < width) & (ys < height)
        indices = (ys[valid] * width + xs[valid]).tolist()

        values = gray.ravel().tolist()
        out = [0.0] * len(values)
        error = 0.0
        for idx in indices:
            value = values[idx] + error
            new_pixel = 1.0 if value > threshold else 0.0
            out[idx] = new_pixel
            error = value - new_pixel

        return np.array(out).reshape(height, width)

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

            # Interpolate so fractional (temporally blended) values stay valid
            d = dithered[..., None]
            return shadow_rgb * (1.0 - d) + highlight_rgb * d

        elif color_mode == "tritone":
            # Three colors with midtones
            shadow_rgb = np.array(self._hex_to_rgb(shadow_color))
            midtone_rgb = np.array(self._hex_to_rgb(midtone_color))
            highlight_rgb = np.array(self._hex_to_rgb(highlight_color))

            # Adjust brightness
            shadow_rgb = np.clip(shadow_rgb + shadow_brightness, 0, 1)
            midtone_rgb = np.clip(midtone_rgb + midtone_brightness, 0, 1)
            highlight_rgb = np.clip(highlight_rgb + highlight_brightness, 0, 1)

            # Bright pixels alternate between highlight and midtone on a
            # checkerboard; interpolate from shadow for fractional values.
            checker = ((np.arange(height)[:, None] + np.arange(width)[None, :]) % 2).astype(bool)
            bright_rgb = np.where(checker[..., None], highlight_rgb, midtone_rgb)

            d = dithered[..., None]
            return shadow_rgb * (1.0 - d) + bright_rgb * d

        else:  # monochrome fallback
            return np.stack([dithered, dithered, dithered], axis=-1)

    def _apply_indexed_mapping(self, dithered, rgb, shadow_color, shadow_brightness, palette_colors):
        """Map dither pattern onto the source colors and quantize to an indexed palette"""
        shadow_rgb = np.clip(np.array(self._hex_to_rgb(shadow_color)) + shadow_brightness, 0, 1)
        d = dithered[..., None]
        colored = np.clip(shadow_rgb * (1.0 - d) + rgb * d, 0, 1)

        pil = Image.fromarray((colored * 255.0).astype(np.uint8), mode="RGB")
        quantized = pil.quantize(colors=int(palette_colors)).convert("RGB")
        return np.asarray(quantized).astype(np.float64) / 255.0

    def _special_dither(self, image, algorithm, threshold=0.5, effect_size=1.0, rng=None):
        """Handle special dithering algorithms"""
        if rng is None:
            rng = np.random.default_rng(0)

        height, width = image.shape[:2]
        gray = self._to_gray(image)

        output = np.zeros_like(gray)

        if algorithm == "random":
            # Random dithering (vectorized)
            random_thresholds = rng.random((height, width)) * effect_size
            output = (gray > random_thresholds).astype(np.float64)

        elif algorithm == "white_noise":
            # Uniform random threshold per pixel
            output = (gray > rng.random((height, width))).astype(np.float64)

        elif algorithm == "threshold":
            # Simple threshold
            output = (gray > threshold).astype(float)

        elif algorithm == "checkerboard":
            # Alternating threshold cells
            cell = max(1, int(round(2 * effect_size)))
            yy, xx = np.mgrid[0:height, 0:width]
            checker = ((yy // cell + xx // cell) % 2).astype(np.float64)
            output = (gray > (0.25 + 0.5 * checker)).astype(np.float64)

        elif algorithm == "spiral":
            # Spiral threshold pattern per cell
            cell = max(4, int(round(8 * effect_size)))
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
            cy = (yy % cell) - (cell - 1) / 2.0
            cx = (xx % cell) - (cell - 1) / 2.0
            radius = np.sqrt(cy * cy + cx * cx) / (cell / 2.0)
            angle = np.arctan2(cy, cx) / (2 * np.pi) + 0.5
            output = (gray > (radius + angle) % 1.0).astype(np.float64)

        elif algorithm == "zigzag":
            # Zigzag threshold ramps
            period = max(2, int(round(4 * effect_size)))
            yy, xx = np.mgrid[0:height, 0:width]
            zig = np.abs((yy % (2 * period)) - period)
            output = (gray > ((xx + zig) % period) / period).astype(np.float64)

        elif algorithm == "halftone_lines":
            # Horizontal line screen: line thickness follows brightness
            spacing = max(2, int(round(4 * effect_size)))
            yy = np.mgrid[0:height, 0:width][0]
            phase = (yy % spacing) / spacing
            line_threshold = np.abs(phase - 0.5) * 2.0
            output = (gray > line_threshold).astype(np.float64)

        elif algorithm == "halftone_dots":
            # Simulate halftone dots (vectorized: block means + distance grid)
            cell = max(1, int(4 * effect_size))
            blocks_y = (height + cell - 1) // cell
            blocks_x = (width + cell - 1) // cell
            padded = np.pad(gray, ((0, blocks_y * cell - height), (0, blocks_x * cell - width)), mode='edge')
            means = padded.reshape(blocks_y, cell, blocks_x, cell).mean(axis=(1, 3))

            radius = np.floor(cell * means / 2.0)
            offsets = np.arange(cell) - cell // 2
            dist2 = offsets[:, None] ** 2 + offsets[None, :] ** 2
            mask = dist2[None, None, :, :] <= (radius ** 2)[:, :, None, None]
            output = mask.astype(np.float64).transpose(0, 2, 1, 3).reshape(blocks_y * cell, blocks_x * cell)[:height, :width]

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
            from scipy.ndimage import uniform_filter
            line_spacing = max(1, int(5 / effect_size))  # Controls line density
            smoothed = uniform_filter(gray, size=7, mode='nearest')

            # First pass: create base line pattern
            for y in range(0, height, line_spacing):
                output[y] = (gray[y] > threshold).astype(np.float64)

            # Second pass: modulate the lines based on image content
            for y in range(0, height, line_spacing):
                # Track the line position with modulation
                prev_offset = 0

                for x in range(width):
                    local_brightness = smoothed[y, x]

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
               midtone_brightness=0.5, highlight_brightness=1.0, palette_colors=2, seed=0):
        """Main dithering function with batch processing support and progress bar"""

        # Convert from torch tensor to numpy array
        if isinstance(image, torch.Tensor):
            device = image.device
            batch_np = image.cpu().numpy()
        else:
            device = torch.device("cpu")
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
            frame = batch_np[i].astype(np.float64)
            if frame.ndim == 2:
                frame = frame[..., None]

            # Split channels: process RGB, pass alpha through unchanged
            channels = frame.shape[-1]
            alpha = frame[..., 3:4] if channels == 4 else None
            if channels == 1:
                rgb = np.repeat(frame, 3, axis=-1)
            else:
                rgb = frame[..., :3]

            rng = np.random.default_rng(seed + i)

            # Apply preprocessing
            rgb = self._apply_preprocessing(rgb, sharpen, blur, noise, rng)

            # Apply dithering algorithm
            if algorithm in self.error_matrices:
                dithered = self._error_diffusion_dither(rgb, algorithm, threshold)
            elif "ordered" in algorithm or "bayer" in algorithm:
                dithered = self._ordered_dither(rgb, algorithm, threshold, effect_size)
            elif algorithm in self.halftone_matrices:
                dithered = self._matrix_dither(self._to_gray(rgb), self.halftone_matrices[algorithm], effect_size)
            elif algorithm in ("blue_noise", "void_and_cluster"):
                matrix = self._get_void_cluster_matrix(32 if algorithm == "blue_noise" else 16)
                dithered = self._matrix_dither(self._to_gray(rgb), matrix, effect_size)
            elif algorithm == "hilbert_curve":
                dithered = self._hilbert_dither(rgb, threshold)
            else:
                dithered = self._special_dither(rgb, algorithm, threshold, effect_size, rng)

            # Apply temporal coherence if enabled and not the first frame
            if temporal_coherence > 0 and previous_dithered is not None and i > 0:
                # Blend with previous frame to reduce flickering
                dithered = (1 - temporal_coherence) * dithered + temporal_coherence * previous_dithered

            previous_dithered = dithered.copy()

            # Apply color mapping
            if color_mode == "indexed":
                result = self._apply_indexed_mapping(dithered, rgb, shadow_color, shadow_brightness, palette_colors)
            else:
                result = self._apply_color_mapping(
                    dithered, color_mode, shadow_color, midtone_color, highlight_color,
                    shadow_brightness, midtone_brightness, highlight_brightness
                )

            if alpha is not None:
                result = np.concatenate([result, alpha], axis=-1)

            results.append(result)
            pbar.update(1)

        # Stack all results into a batch
        batch_result = np.stack(results, axis=0)

        # Convert back to torch tensor
        result_tensor = torch.from_numpy(batch_result).float().clamp(0.0, 1.0).to(device)

        return (result_tensor,)
