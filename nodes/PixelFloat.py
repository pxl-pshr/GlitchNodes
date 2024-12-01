import torch
import numpy as np
import cv2
from tqdm import tqdm

class PixelFloat:
    def __init__(self):
        self.old_mvs = None
        self.rt = 0
        
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
    FUNCTION = "process_frames"
    CATEGORY = "image/animation"

    def ensure_rgb(self, image):
        if len(image.shape) == 3:
            if image.shape[-1] != 3:
                image = np.transpose(image, (1, 2, 0))
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.shape[-1] > 3:
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
        
        return best_block_size

    def estimate_motion_vectors(self, frame1, frame2, flow_scale, flow_levels, flow_iterations, block_size, auto_block_size):
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
        
        if auto_block_size:
            actual_block_size = self.calculate_block_size(
                frame1.shape[0], 
                frame1.shape[1],
                self.min_blocks,
                self.max_blocks
            )
        else:
            h, w = frame1.shape[:2]
            if h % block_size != 0 or w % block_size != 0:
                valid_h = h // (h // block_size)
                valid_w = w // (w // block_size)
                actual_block_size = min(valid_h, valid_w)
            else:
                actual_block_size = block_size
        
        h, w = flow.shape[:2]
        blocks_h = h // actual_block_size
        blocks_w = w // actual_block_size
        
        mvs = []
        for i in range(blocks_h):
            row = []
            for j in range(blocks_w):
                block = flow[i*actual_block_size:(i+1)*actual_block_size, 
                           j*actual_block_size:(j+1)*actual_block_size]
                mv = np.mean(block, axis=(0,1))
                
                # Calculate block variance to detect static areas
                block_var = np.var(block)
                if block_var < 0.01:  # Threshold for static detection
                    mv *= 0.1  # Reduce motion for static areas
                
                row.append(mv.tolist())
            mvs.append(row)
            
        return mvs, actual_block_size

    def apply_anti_gravity(self, mvs, gravity_strength, motion_threshold):
        if self.rt == 0:
            self.old_mvs = [[[mv[:] for mv in row] for row in mvs]]
            self.rt = 1
            return mvs
        
        old_mvs = self.old_mvs[0]
        
        if len(mvs) != len(old_mvs) or len(mvs[0]) != len(old_mvs[0]):
            self.old_mvs = [[[mv[:] for mv in row] for row in mvs]]
            return mvs
        
        # Smooth motion vectors between neighboring blocks
        smoothed_mvs = [[mv[:] for mv in row] for row in mvs]
        for i in range(len(mvs)):
            for j in range(len(mvs[0])):
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < len(mvs) and 0 <= nj < len(mvs[0]):
                            neighbors.append(mvs[ni][nj])
                
                avg_motion = np.mean(neighbors, axis=0)
                current_motion = mvs[i][j]
                
                # Only apply smoothing if motion difference is above threshold
                motion_diff = np.linalg.norm(np.array(current_motion) - avg_motion)
                if motion_diff > motion_threshold:
                    smoothed_mvs[i][j] = (0.7 * np.array(current_motion) + 0.3 * avg_motion).tolist()
        
        # Apply gravity effect with smoothed vectors
        for i in range(len(smoothed_mvs)):
            for j in range(len(smoothed_mvs[0])):
                mv = smoothed_mvs[i][j]
                omv = old_mvs[i][j]
                
                if mv[1] < 0:
                    nmv = mv[1]
                    mv[1] = omv[1]
                    omv[1] = nmv + omv[1] - gravity_strength
                    
                    # Dampen extreme motions
                    if abs(mv[1]) > 5 * abs(gravity_strength):
                        mv[1] *= 0.8
        
        return smoothed_mvs

    def apply_motion_vectors(self, frame, mvs, block_size, interpolation_factor):
        h, w = frame.shape[:2]
        blocks_h = h // block_size
        blocks_w = w // block_size
        
        # Create interpolated grid for smoother transitions
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Create displacement maps
        displacement_y = np.zeros((h, w), dtype=np.float32)
        displacement_x = np.zeros((h, w), dtype=np.float32)
        
        for i in range(blocks_h):
            for j in range(blocks_w):
                mv = mvs[i][j]
                
                # Apply motion vectors to block
                block_y = slice(i*block_size, (i+1)*block_size)
                block_x = slice(j*block_size, (j+1)*block_size)
                
                displacement_y[block_y, block_x] = mv[1]
                displacement_x[block_y, block_x] = mv[0]
        
        # Apply Gaussian blur to displacement maps for smoother transitions
        displacement_y = cv2.GaussianBlur(displacement_y, (5, 5), 1.5)
        displacement_x = cv2.GaussianBlur(displacement_x, (5, 5), 1.5)
        
        # Apply interpolation factor
        y += displacement_y * interpolation_factor
        x += displacement_x * interpolation_factor
        
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
        
        batch_size = frames_np.shape[0]
        processed_frames = []
        
        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        
        # Simplified progress bar configuration
        pbar = tqdm(range(batch_size-1), 
                   desc="Processing frames",
                   leave=True)
        
        try:
            for i in pbar:
                current_frame = frames_np[i].copy()
                next_frame = frames_np[i+1].copy()
                
                current_frame = (current_frame * 255).astype(np.uint8)
                next_frame = (next_frame * 255).astype(np.uint8)
                
                mvs, actual_block_size = self.estimate_motion_vectors(
                    current_frame, 
                    next_frame,
                    flow_scale,
                    flow_levels,
                    flow_iterations,
                    block_size,
                    auto_block_size
                )
                modified_mvs = self.apply_anti_gravity(mvs, gravity_strength, motion_threshold)
                processed_frame = self.apply_motion_vectors(
                    current_frame, 
                    modified_mvs, 
                    actual_block_size,
                    interpolation_factor
                )
                
                processed_frame = processed_frame.astype(np.float32) / 255.0
                processed_frames.append(processed_frame)
                
                # Update progress description with current frame
                pbar.set_description(f"Processing frame {i+1}/{batch_size-1}")
                
            pbar.close()
            
        except Exception as e:
            pbar.close()
            raise RuntimeError(f"Error processing frame {i}: {str(e)}")
        
        processed_frames.append(frames_np[-1])
        processed_frames = np.stack(processed_frames)
        return (torch.from_numpy(processed_frames).to(frames.device),)