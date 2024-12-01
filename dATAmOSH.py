import cv2
import numpy as np
import torch
from tqdm import tqdm

class dATAmOSH:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video": ("IMAGE",),
                "start_frame": ("INT", {"default": 0, "min": 0, "step": 1}),
                "pyr_scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.1}),
                "levels": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
                "winsize": ("INT", {"default": 15, "min": 3, "max": 51, "step": 2}),
                "iterations": ("INT", {"default": 3, "min": 1, "max": 15, "step": 1}),
                "poly_n": ("INT", {"default": 5, "min": 3, "max": 7, "step": 2}),
                "poly_sigma": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "target_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transfer_optical_flow"
    CATEGORY = "image/animation"

    def transfer_optical_flow(self, source_video, start_frame, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, target_image=None):
        # Convert source_video tensor to numpy array
        video_frames = source_video.cpu().numpy()
        num_frames, height, width, depth = video_frames.shape
        
        # Ensure start_frame is within bounds
        start_frame = min(max(start_frame, 0), num_frames - 1)
        
        # Initialize output_frames with original frames
        output_frames = video_frames.copy()
        
        # Use the frame at start_frame as the initial target image if no target image is provided
        if target_image is None:
            target_image_np = video_frames[start_frame]
        else:
            target_image_np = target_image.squeeze().cpu().numpy()
        
        # Ensure target_image_np has the correct shape
        if target_image_np.shape != (height, width, depth):
            target_image_np = cv2.resize(target_image_np, (width, height))
        
        # Initialize image_frame for datamosh effect
        image_frame = (target_image_np * 255).astype(np.uint8)

        base = np.arange(height * width * depth, dtype=int)
        
        prev_gray = cv2.cvtColor((video_frames[start_frame] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Create progress bar
        with tqdm(total=num_frames-start_frame, desc="DATAMOSHING", unit="frame") as pbar:
            for i in range(start_frame, num_frames):
                if i == start_frame:
                    output_frames[i] = image_frame / 255.0  # Normalize back to 0-1 range
                else:
                    gray = cv2.cvtColor((video_frames[i] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(
                        prev=prev_gray,
                        next=gray,
                        flow=None,
                        pyr_scale=pyr_scale,
                        levels=levels,
                        winsize=winsize,
                        iterations=iterations,
                        poly_n=poly_n,
                        poly_sigma=poly_sigma, 
                        flags=0
                    ).astype(int)
                    
                    flow_flat = np.repeat(
                        flow[:, :, 1] * width * depth + flow[:, :, 0] * depth,
                        depth
                    )
                    
                    np.put(image_frame, base + flow_flat, image_frame.flat, mode="wrap")
                    output_frames[i] = image_frame / 255.0  # Normalize back to 0-1 range
                    
                    prev_gray = gray

                pbar.update(1)

        # Convert output frames back to PyTorch tensor
        output_tensor = torch.from_numpy(output_frames).float()
        output_tensor = output_tensor.to(source_video.device)
        
        return (output_tensor,)