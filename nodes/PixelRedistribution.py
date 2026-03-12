# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import logging
import torch
import numpy as np
import comfy.utils
import math
from collections import deque

logger = logging.getLogger(__name__)

class PixelRedistribution:
    """Redistribute pixels based on color distance and pattern."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "distance_mode": (["center", "top", "left", "random"], {"default": "center"}),
                "pattern": (["outward", "spiral", "waves", "diagonal"], {"default": "outward"}),
                "color_size": ("INT", {
                    "default": 64,
                    "min": 2,
                    "max": 256,
                    "step": 1
                }),
                "order": ("STRING", {"default": "0,1,2"}),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 4.0,
                    "step": 0.1
                }),
                "brightness": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "invert": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "redistribute_pixels"
    CATEGORY = "GlitchNodes"
    DESCRIPTION = "Redistributes pixels based on color distance and pattern modes"

    def adjust_contrast_brightness(self, image, contrast, brightness):
        mean = image.mean()
        adjusted = (image - mean) * contrast + mean
        if brightness > 0:
            adjusted = adjusted * (1 - brightness) + brightness
        else:
            adjusted = adjusted * (1 + brightness)
        return np.clip(adjusted, 0, 1)

    def calculate_distance_np(self, coords, width, height, mode, pattern, strength):
        """Vectorized distance calculation using numpy arrays.
        coords: (N, 2) array with (x, y) columns."""
        x = coords[:, 0].astype(np.float64)
        y = coords[:, 1].astype(np.float64)

        if mode == "center":
            base_distance = np.sqrt((x - width / 2) ** 2 + (y - height / 2) ** 2)
        elif mode == "top":
            base_distance = y.copy()
        elif mode == "left":
            base_distance = x.copy()
        elif mode == "random":
            base_distance = np.random.rand(len(x))
        else:
            base_distance = np.zeros(len(x))

        if pattern == "spiral":
            angle = np.arctan2(y - height / 2, x - width / 2)
            base_distance = base_distance + angle * width / (2 * math.pi)
        elif pattern == "waves":
            wave = np.sin(x * 0.1) * height * 0.25
            base_distance = base_distance + wave
        elif pattern == "diagonal":
            base_distance = base_distance + (x + y) * 0.5

        return base_distance * strength

    def get_adjacent_colors_np(self, color, searched, order, color_size):
        """Get adjacent colors using plain tuples — no torch overhead."""
        adj_list = []
        for channel in order:
            if color[channel] > 0:
                new_color = list(color)
                new_color[channel] -= 1
                new_color = tuple(new_color)
                if new_color not in searched:
                    adj_list.append(new_color)
            if color[channel] < color_size - 1:
                new_color = list(color)
                new_color[channel] += 1
                new_color = tuple(new_color)
                if new_color not in searched:
                    adj_list.append(new_color)
        return adj_list

    def process_single_image(self, image_np, color_size, order, distance_mode, pattern, strength, contrast, brightness, invert):
        """Process a single [H, W, C] numpy image."""
        height, width, channels = image_np.shape

        # Adjust contrast and brightness
        process_image = self.adjust_contrast_brightness(image_np, contrast, brightness)
        if invert:
            process_image = 1.0 - process_image

        # Quantize to color_size levels
        quantized = np.floor(process_image * (color_size - 1)).astype(np.int32)

        # Group pixels by quantized color — vectorized
        # Encode each pixel's color as a single int for fast grouping
        # color key = r * cs^2 + g * cs + b (unique for each color combo)
        cs = color_size
        if channels >= 3:
            color_keys = quantized[:, :, 0] * cs * cs + quantized[:, :, 1] * cs + quantized[:, :, 2]
        else:
            color_keys = quantized[:, :, 0]

        # Build mapping: color_key -> list of (x, y) coords
        flat_keys = color_keys.ravel()
        ys, xs = np.mgrid[0:height, 0:width]
        flat_x = xs.ravel()
        flat_y = ys.ravel()

        # Use numpy sorting to group by color key
        sort_idx = np.argsort(flat_keys, kind='mergesort')
        sorted_keys = flat_keys[sort_idx]
        sorted_x = flat_x[sort_idx]
        sorted_y = flat_y[sort_idx]

        # Find boundaries between groups
        boundaries = np.where(np.diff(sorted_keys) != 0)[0] + 1
        boundaries = np.concatenate([[0], boundaries, [len(sorted_keys)]])

        # Build the color -> coords dict
        transform_colorspace = {}
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            key_val = sorted_keys[start]
            # Decode back to color tuple
            if channels >= 3:
                r = int(key_val // (cs * cs))
                g = int((key_val % (cs * cs)) // cs)
                b = int(key_val % cs)
                color = (r, g, b)
            else:
                color = (int(key_val),)
            coords = np.column_stack([sorted_x[start:end], sorted_y[start:end]])
            transform_colorspace[color] = coords  # (N, 2) array of (x, y)

        # Sort pixels within each multi-pixel color group by distance
        start_keys = sorted(
            [k for k, v in transform_colorspace.items() if len(v) > 1],
            key=lambda x: len(transform_colorspace[x])
        )

        for start_key in start_keys:
            coords = transform_colorspace[start_key]
            distances = self.calculate_distance_np(coords, width, height, distance_mode, pattern, strength)
            sorted_indices = np.argsort(distances)
            transform_colorspace[start_key] = coords[sorted_indices]

            # BFS to find neighboring empty colors to redistribute excess pixels
            queue = deque()
            searched = set()
            prev = {}
            end_keys = []

            queue.append(start_key)
            searched.add(start_key)
            prev[start_key] = None

            target_count = len(transform_colorspace[start_key]) - 1

            while queue and len(end_keys) < target_count:
                current_key = queue.popleft()

                if current_key not in transform_colorspace or len(transform_colorspace[current_key]) == 0:
                    end_keys.append(current_key)
                    if len(end_keys) >= target_count:
                        break

                adj_colors = self.get_adjacent_colors_np(current_key, searched, order, color_size)
                for adj_key in adj_colors:
                    searched.add(adj_key)
                    prev[adj_key] = current_key
                    queue.append(adj_key)

            # Redistribute pixels along BFS paths
            for end_key in end_keys:
                if end_key not in prev or prev[end_key] is None:
                    continue
                current_key = end_key
                while prev.get(current_key) is not None:
                    prev_key = prev[current_key]
                    if prev_key in transform_colorspace and len(transform_colorspace[prev_key]) > 0:
                        # Pop first point from source
                        src = transform_colorspace[prev_key]
                        point = src[0:1]
                        transform_colorspace[prev_key] = src[1:]
                        # Append to destination
                        if current_key not in transform_colorspace or len(transform_colorspace[current_key]) == 0:
                            transform_colorspace[current_key] = point
                        else:
                            transform_colorspace[current_key] = np.vstack([transform_colorspace[current_key], point])
                    current_key = prev_key

        # Build output image — vectorized
        output = np.zeros_like(image_np)
        for color, pts in transform_colorspace.items():
            if len(pts) == 0:
                continue
            color_arr = np.array(color, dtype=np.float32) / (color_size - 1)
            if invert:
                color_arr = 1.0 - color_arr
            if isinstance(pts, np.ndarray):
                xs_out = pts[:, 0].astype(np.int32)
                ys_out = pts[:, 1].astype(np.int32)
            else:
                xs_out = np.array([p[0] for p in pts], dtype=np.int32)
                ys_out = np.array([p[1] for p in pts], dtype=np.int32)
            output[ys_out, xs_out, :len(color_arr)] = color_arr

        return output

    def redistribute_pixels(self, image, color_size, order, distance_mode, pattern, strength, contrast, brightness, invert):
        try:
            if not isinstance(image, torch.Tensor):
                raise ValueError("Input image must be a torch.Tensor")

            if image.dim() != 4:
                raise ValueError(f"Expected 4D input tensor, got {image.dim()}D")

            try:
                order = [int(x.strip()) for x in order.split(',')]
                if not all(0 <= x <= 2 for x in order) or len(order) != 3:
                    raise ValueError("Order must be three comma-separated integers between 0 and 2")
            except ValueError as e:
                raise ValueError(f"Invalid order format: {e}")

            # Convert to numpy once for the whole batch
            image_np = image.cpu().numpy().astype(np.float32)
            if image_np.max() > 1.0:
                image_np = image_np / 255.0

            batch_size = image_np.shape[0]
            processed_images = []

            pbar = comfy.utils.ProgressBar(batch_size)
            for i in range(batch_size):
                processed_image = self.process_single_image(
                    image_np[i],
                    color_size,
                    order,
                    distance_mode,
                    pattern,
                    strength,
                    contrast,
                    brightness,
                    invert
                )
                processed_images.append(processed_image)
                pbar.update(1)

            output = np.stack(processed_images, axis=0)
            output = torch.from_numpy(output).float().clamp(0, 1).to(image.device)

            return (output,)

        except Exception as e:
            logger.error(f"Error in PixelRedistribution: {str(e)}")
            raise e
