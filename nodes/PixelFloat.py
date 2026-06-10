# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import logging
import torch
import numpy as np
import cv2
from scipy.ndimage import uniform_filter
import comfy.utils

logger = logging.getLogger(__name__)

class PixelFloat:
    """Apply gravity effects to pixel blocks with motion estimation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "gravity_strength": ("FLOAT", {
                    "default": -10.0,
                    "min": -50.0,
                    "max": 0.0,
                    "step": 0.5
                }),
                "block_size": ("INT", {
                    "default": 4,
                    "min": 4,
                    "max": 64,
                    "step": 4
                }),
                "auto_block_size": ("BOOLEAN", {
                    "default": False,
                }),
                "min_blocks": ("INT", {
                    "default": 32,
                    "min": 16,
                    "max": 64,
                    "step": 4
                }),
                "max_blocks": ("INT", {
                    "default": 128,
                    "min": 64,
                    "max": 256,
                    "step": 4
                }),
                "flow_scale": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05
                }),
                "flow_levels": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 8,
                    "step": 1
                }),
                "flow_iterations": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "motion_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                }),
                "interpolation_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_frames"
    CATEGORY = "GlitchNodes"
    DESCRIPTION = "Applies gravity effects to pixel blocks based on optical flow motion estimation"

    def ensure_rgb(self, image):
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] > 3:
            image = image[:, :, :3]
        return image

    def calculate_block_size(self, height, width, min_blocks, max_blocks):
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        common_factors = []
        gcd_val = gcd(height, width)
        for i in range(1, int(np.sqrt(gcd_val)) + 1):
            if gcd_val % i == 0:
                common_factors.append(i)
                if i != gcd_val // i:
                    common_factors.append(gcd_val // i)

        common_factors.sort()

        best_block_size = 16
        min_diff = float('inf')
        target_blocks = (min_blocks + max_blocks) // 2

        for factor in common_factors:
            h_blocks = height // factor
            w_blocks = width // factor

            if min_blocks <= h_blocks <= max_blocks and min_blocks <= w_blocks <= max_blocks:
                diff = abs(target_blocks - h_blocks) + abs(target_blocks - w_blocks)
                if diff < min_diff:
                    min_diff = diff
                    best_block_size = factor

        return max(1, min(best_block_size, height, width))

    def estimate_motion_vectors(self, frame1, frame2, flow_scale, flow_levels, flow_iterations, block_size, auto_block_size, min_blocks, max_blocks):
        frame1 = self.ensure_rgb(frame1)
        frame2 = self.ensure_rgb(frame2)

        # Apply Gaussian blur to reduce noise
        frame1_blur = cv2.GaussianBlur(frame1, (3, 3), 0)
        frame2_blur = cv2.GaussianBlur(frame2, (3, 3), 0)

        f1 = cv2.cvtColor(frame1_blur, cv2.COLOR_RGB2GRAY)
        f2 = cv2.cvtColor(frame2_blur, cv2.COLOR_RGB2GRAY)

        if f1.shape != f2.shape:
            f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))

        try:
            flow = cv2.calcOpticalFlowFarneback(
                f1, f2,
                None,
                pyr_scale=flow_scale,
                levels=flow_levels,
                winsize=15,
                iterations=flow_iterations,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
        except cv2.error as e:
            raise RuntimeError(f"Optical flow error: {str(e)}\nShapes: f1={f1.shape}, f2={f2.shape}")

        h, w = flow.shape[:2]
        if auto_block_size:
            actual_block_size = self.calculate_block_size(
                frame1.shape[0],
                frame1.shape[1],
                min_blocks,
                max_blocks
            )
        else:
            actual_block_size = block_size
        actual_block_size = max(1, min(actual_block_size, h, w))

        blocks_h = h // actual_block_size
        blocks_w = w // actual_block_size

        blocks = flow[:blocks_h * actual_block_size, :blocks_w * actual_block_size].reshape(
            blocks_h, actual_block_size, blocks_w, actual_block_size, 2)
        mvs = blocks.mean(axis=(1, 3)).astype(np.float32)

        # Reduce motion for static areas detected via block variance
        block_var = blocks.var(axis=(1, 3, 4))
        mvs[block_var < 0.01] *= 0.1

        return mvs, actual_block_size

    def apply_anti_gravity(self, mvs, gravity_strength, motion_threshold, carry):
        if carry is None or carry.shape != mvs.shape[:2]:
            carry = np.zeros(mvs.shape[:2], dtype=np.float32)

        # Smooth motion vectors between neighboring blocks
        avg_motion = np.stack(
            [uniform_filter(mvs[..., k], size=3, mode='nearest') for k in range(2)],
            axis=-1
        )
        smoothed_mvs = mvs.copy()
        motion_diff = np.linalg.norm(mvs - avg_motion, axis=-1)
        smooth_mask = motion_diff > motion_threshold
        smoothed_mvs[smooth_mask] = 0.7 * mvs[smooth_mask] + 0.3 * avg_motion[smooth_mask]

        # Apply gravity effect: rising blocks use the accumulated displacement,
        # which keeps growing by -gravity_strength every frame
        rising = smoothed_mvs[..., 1] < 0
        new_carry = np.where(
            rising,
            smoothed_mvs[..., 1] + carry - gravity_strength,
            carry
        ).astype(np.float32)

        out_mvs = smoothed_mvs.copy()
        out_mvs[..., 1] = np.where(rising, carry, smoothed_mvs[..., 1])

        # Dampen extreme motions
        extreme = rising & (np.abs(out_mvs[..., 1]) > 5 * abs(gravity_strength))
        out_mvs[..., 1] = np.where(extreme, out_mvs[..., 1] * 0.8, out_mvs[..., 1])

        return out_mvs, new_carry

    def apply_motion_vectors(self, frame, mvs, block_size, interpolation_factor):
        h, w = frame.shape[:2]

        # Expand block motion vectors to per-pixel displacement maps,
        # padding any remainder strip with the nearest block's displacement
        displacement = np.repeat(np.repeat(mvs, block_size, axis=0), block_size, axis=1)
        displacement = np.pad(
            displacement,
            ((0, h - displacement.shape[0]), (0, w - displacement.shape[1]), (0, 0)),
            mode='edge'
        ).astype(np.float32)

        # Apply Gaussian blur to displacement maps for smoother transitions
        displacement_x = cv2.GaussianBlur(displacement[..., 0], (5, 5), 1.5)
        displacement_y = cv2.GaussianBlur(displacement[..., 1], (5, 5), 1.5)

        y, x = np.mgrid[0:h, 0:w].astype(np.float32)

        # Apply interpolation factor
        x += displacement_x * interpolation_factor
        y += displacement_y * interpolation_factor

        # Ensure coordinates stay within bounds
        x = np.clip(x, 0, w-1)
        y = np.clip(y, 0, h-1)

        # Use cubic interpolation for smoother results
        output = cv2.remap(frame, x, y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        return output

    def process_frames(self, frames, gravity_strength, block_size, auto_block_size,
                      min_blocks, max_blocks, flow_scale, flow_levels, flow_iterations,
                      motion_threshold, interpolation_factor):
        frames_np = frames.cpu().numpy()

        if len(frames_np.shape) == 3:
            frames_np = np.expand_dims(frames_np, 0)

        alpha = None
        if frames_np.shape[-1] == 4:
            alpha = frames_np[..., 3:]
            frames_np = frames_np[..., :3]
        elif frames_np.shape[-1] == 1:
            frames_np = np.repeat(frames_np, 3, axis=-1)

        batch_size = frames_np.shape[0]
        processed_frames = []

        # Create progress bar
        pbar = comfy.utils.ProgressBar(max(batch_size, 1))

        carry = None
        last_mvs = None
        last_block_size = None
        frame_idx = 'unknown'

        try:
            for i in range(batch_size - 1):
                frame_idx = i
                current_frame = (frames_np[i] * 255).astype(np.uint8)
                next_frame = (frames_np[i+1] * 255).astype(np.uint8)

                mvs, actual_block_size = self.estimate_motion_vectors(
                    current_frame,
                    next_frame,
                    flow_scale,
                    flow_levels,
                    flow_iterations,
                    block_size,
                    auto_block_size,
                    min_blocks,
                    max_blocks
                )
                modified_mvs, carry = self.apply_anti_gravity(mvs, gravity_strength, motion_threshold, carry)
                processed_frame = self.apply_motion_vectors(
                    current_frame,
                    modified_mvs,
                    actual_block_size,
                    interpolation_factor
                )

                processed_frame = processed_frame.astype(np.float32) / 255.0
                processed_frames.append(processed_frame)

                last_mvs = mvs
                last_block_size = actual_block_size

                pbar.update(1)

            # Process the last frame with the last motion vectors to avoid a visible pop
            frame_idx = batch_size - 1
            last_frame = (frames_np[-1] * 255).astype(np.uint8)
            if last_mvs is not None:
                modified_mvs, carry = self.apply_anti_gravity(last_mvs, gravity_strength, motion_threshold, carry)
                last_frame = self.apply_motion_vectors(
                    last_frame,
                    modified_mvs,
                    last_block_size,
                    interpolation_factor
                )
            processed_frames.append(last_frame.astype(np.float32) / 255.0)
            pbar.update(1)

        except Exception as e:
            logger.error(f"Error processing frame {frame_idx}: {str(e)}")
            raise RuntimeError(f"Error processing frame {frame_idx}: {str(e)}")

        output = np.stack(processed_frames)
        if alpha is not None:
            output = np.concatenate([output, alpha], axis=-1)
        return (torch.from_numpy(output).float().clamp(0, 1).to(frames.device),)
