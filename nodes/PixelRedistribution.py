# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import logging
import torch
import numpy as np
import comfy.utils
import math

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

    def calculate_distance_map(self, width, height, mode, pattern, strength):
        """Compute spatial distance for every pixel — fully vectorized."""
        ys, xs = np.mgrid[0:height, 0:width]
        x = xs.astype(np.float64)
        y = ys.astype(np.float64)

        if mode == "center":
            base = np.sqrt((x - width / 2) ** 2 + (y - height / 2) ** 2)
        elif mode == "top":
            base = y.copy()
        elif mode == "left":
            base = x.copy()
        elif mode == "random":
            base = np.random.rand(height, width)
        else:
            base = np.zeros((height, width))

        if pattern == "spiral":
            angle = np.arctan2(y - height / 2, x - width / 2)
            base = base + angle * width / (2 * math.pi)
        elif pattern == "waves":
            base = base + np.sin(x * 0.1) * height * 0.25
        elif pattern == "diagonal":
            base = base + (x + y) * 0.5

        return base * strength

    def process_single_image(self, image_np, color_size, order, distance_mode, pattern, strength, contrast, brightness, invert):
        """Process a single [H, W, C] numpy image — fully vectorized."""
        height, width, channels = image_np.shape
        n_ch = min(channels, 3)
        cs = color_size

        # Adjust contrast and brightness
        process_image = self.adjust_contrast_brightness(image_np, contrast, brightness)
        if invert:
            process_image = 1.0 - process_image

        # Quantize to color_size levels
        quantized = np.floor(process_image[:, :, :n_ch] * (cs - 1)).astype(np.int32)

        # Spatial distance map — pixels far from the origin are more likely to shift color
        dist_map = self.calculate_distance_map(width, height, distance_mode, pattern, strength)
        d_max = dist_map.max()
        norm_dist = dist_map / d_max if d_max > 0 else np.zeros_like(dist_map)

        # --- Vectorized diffusion: pixels flow from dense to sparse colors ---
        # Far-from-origin pixels move first, creating the spatial pattern.
        n_passes = max(1, int(strength * 5))
        dist_threshold = max(0.0, 1.0 - strength * 0.5)
        current_q = quantized.copy()

        for pass_num in range(n_passes):
            q0 = current_q[:, :, 0]
            q1 = current_q[:, :, 1] if n_ch >= 2 else q0
            q2 = current_q[:, :, 2] if n_ch >= 3 else q0

            # Population grid via bincount
            enc = q0 * cs * cs + q1 * cs + q2 if n_ch >= 3 else q0
            pop_grid_flat = np.bincount(enc.ravel(), minlength=cs ** n_ch)
            if n_ch >= 3:
                pop_grid = pop_grid_flat.reshape(cs, cs, cs)
                pixel_pop = pop_grid[q0, q1, q2]
            else:
                pop_grid = pop_grid_flat
                pixel_pop = pop_grid[q0]

            # Compute populations of all adjacent colors (per channel order × ±1)
            adj_pops = []
            adj_channels = []
            adj_deltas = []

            for ch in order:
                if ch >= n_ch:
                    continue
                for delta in [1, -1]:
                    shifted_ch = np.clip(current_q[:, :, ch] + delta, 0, cs - 1)
                    if n_ch >= 3:
                        if ch == 0:
                            ap = pop_grid[shifted_ch, q1, q2]
                        elif ch == 1:
                            ap = pop_grid[q0, shifted_ch, q2]
                        else:
                            ap = pop_grid[q0, q1, shifted_ch]
                    else:
                        ap = pop_grid[shifted_ch]
                    adj_pops.append(ap)
                    adj_channels.append(ch)
                    adj_deltas.append(delta)

            if not adj_pops:
                break

            # Best direction: least populated adjacent color
            adj_stack = np.stack(adj_pops, axis=-1)
            best_dir_idx = adj_stack.argmin(axis=-1)
            min_adj_pop = np.take_along_axis(
                adj_stack, best_dir_idx[..., np.newaxis], axis=-1
            ).squeeze(-1)

            # Move pixels that are: (a) in oversized groups, (b) far from distance origin
            should_move = (pixel_pop > min_adj_pop + 1) & (norm_dist > dist_threshold)

            if not should_move.any():
                break

            # Decode best direction per pixel
            best_ch = np.empty((height, width), dtype=np.int32)
            best_delta = np.empty((height, width), dtype=np.int32)
            for di in range(len(adj_pops)):
                mask = (best_dir_idx == di)
                best_ch[mask] = adj_channels[di]
                best_delta[mask] = adj_deltas[di]

            # Apply color shifts
            new_q = current_q.copy()
            for ch in range(n_ch):
                ch_mask = should_move & (best_ch == ch)
                if ch_mask.any():
                    new_q[:, :, ch] = np.where(
                        ch_mask,
                        np.clip(current_q[:, :, ch] + best_delta, 0, cs - 1),
                        new_q[:, :, ch]
                    )

            current_q = new_q

        # Build output
        output = np.zeros_like(image_np)
        output[:, :, :n_ch] = current_q.astype(np.float32) / (cs - 1)
        if invert:
            output[:, :, :n_ch] = 1.0 - output[:, :, :n_ch]

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
