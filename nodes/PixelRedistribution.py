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
                    "max": 128,
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
                "invert": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
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

    def calculate_distance_map(self, width, height, mode, pattern, strength, rng=None):
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
            base = rng.random((height, width)) if rng is not None else np.zeros((height, width))
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

    def normalize_distance_map(self, dist_map):
        d_min = dist_map.min()
        d_max = dist_map.max()
        if d_max > d_min:
            return (dist_map - d_min) / (d_max - d_min)
        return np.zeros_like(dist_map)

    def process_single_image(self, image_np, color_size, order, norm_dist, strength, contrast, brightness, invert):
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

        # --- Vectorized diffusion: pixels flow from dense to sparse colors ---
        # Far-from-origin pixels move first, creating the spatial pattern.
        n_passes = max(1, int(strength * 5))
        dist_threshold = max(0.0, 1.0 - strength * 0.5)
        current_q = quantized.copy()

        # Preallocate one population grid and reuse it across passes
        pop_grid_flat = np.zeros(cs ** n_ch, dtype=np.int64)
        pop_grid = pop_grid_flat.reshape(cs, cs, cs) if n_ch >= 3 else pop_grid_flat

        for pass_num in range(n_passes):
            q0 = current_q[:, :, 0]
            q1 = current_q[:, :, 1] if n_ch >= 2 else q0
            q2 = current_q[:, :, 2] if n_ch >= 3 else q0

            # Population grid via bincount into the reusable buffer
            enc = q0 * cs * cs + q1 * cs + q2 if n_ch >= 3 else q0
            counts = np.bincount(enc.ravel())
            pop_grid_flat[:] = 0
            pop_grid_flat[:counts.size] = counts
            if n_ch >= 3:
                pixel_pop = pop_grid[q0, q1, q2]
            else:
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

        # Build output, passing any extra channels (e.g. alpha) through unchanged
        output = image_np.copy()
        output[:, :, :n_ch] = current_q.astype(np.float32) / (cs - 1)
        if invert:
            output[:, :, :n_ch] = 1.0 - output[:, :, :n_ch]

        return output

    def redistribute_pixels(self, image, color_size, order, distance_mode, pattern, strength, contrast, brightness, invert, seed):
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

            batch_size, height, width = image_np.shape[:3]
            processed_images = []

            # Distance map is batch-invariant except in random mode
            norm_dist = None
            if distance_mode != "random":
                dist_map = self.calculate_distance_map(width, height, distance_mode, pattern, strength)
                norm_dist = self.normalize_distance_map(dist_map)

            pbar = comfy.utils.ProgressBar(batch_size)
            for i in range(batch_size):
                if distance_mode == "random":
                    rng = np.random.default_rng(seed + i)
                    dist_map = self.calculate_distance_map(width, height, distance_mode, pattern, strength, rng)
                    norm_dist = self.normalize_distance_map(dist_map)

                processed_image = self.process_single_image(
                    image_np[i],
                    color_size,
                    order,
                    norm_dist,
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
