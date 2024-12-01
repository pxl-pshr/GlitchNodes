import torch
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import cv2

class DataBend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                # Slice/Block Controls
                "slice_direction": (["horizontal", "vertical", "both"], {"default": "horizontal"}),
                "slice_min_size": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1}),
                "slice_max_size": ("INT", {"default": 40, "min": 5, "max": 200, "step": 5}),
                "slice_variability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                
                # Color Manipulation
                "channel_shift_mode": (["random", "rgb_split", "hue_shift"], {"default": "random"}),
                "color_intensity": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "rgb_shift_separate": ("BOOLEAN", {"default": False}),
                "preserve_bright_areas": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                
                # Glitch Pattern Controls
                "glitch_types": (["shift", "repeat", "mirror", "noise", "all"], {"default": "all"}),
                "pattern_frequency": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "chaos_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 99999, "step": 1}),
                
                # Distortion Controls
                "wave_distortion": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "compression_artifacts": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "pixel_sorting": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                
                # Control Parameter
                "control_after_generate": (["randomize", "none"], {"default": "none"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_databend"
    CATEGORY = "GlitchNodes"

    def create_slices(self, height, width, params):
        """Create slice indices based on direction and size parameters"""
        slices = []
        if params["slice_direction"] in ["horizontal", "both"]:
            size = np.random.randint(params["slice_min_size"], params["slice_max_size"] + 1)
            for i in range(0, height - size, size):
                if np.random.random() < params["slice_variability"]:
                    slices.append(("h", i, i + size))
                    
        if params["slice_direction"] in ["vertical", "both"]:
            size = np.random.randint(params["slice_min_size"], params["slice_max_size"] + 1)
            for i in range(0, width - size, size):
                if np.random.random() < params["slice_variability"]:
                    slices.append(("v", i, i + size))
                    
        return slices

    def apply_color_shift(self, image, params):
        """Apply color channel manipulation based on mode"""
        result = image.copy()
        
        if params["channel_shift_mode"] == "rgb_split":
            for c in range(3):
                if params["rgb_shift_separate"] or np.random.random() < 0.5:
                    shift = int(params["color_intensity"] * 20)
                    result[..., c] = np.roll(image[..., c], shift, axis=1)
                    
        elif params["channel_shift_mode"] == "hue_shift":
            hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv[..., 0] = (hsv[..., 0] + int(params["color_intensity"] * 180)) % 180
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255
            
        else:  # random
            shifts = np.random.randint(-int(params["color_intensity"] * 30),
                                     int(params["color_intensity"] * 30), 3)
            for c in range(3):
                result[..., c] = np.roll(image[..., c], shifts[c], axis=1)
                
        return result

    def apply_glitch_patterns(self, image, params):
        """Apply various glitch patterns based on type"""
        result = image.copy()
        height, width = image.shape[:2]
        
        for _ in range(params["pattern_frequency"]):
            if params["glitch_types"] == "all":
                effect = np.random.choice(["shift", "repeat", "mirror", "noise"])
            else:
                effect = params["glitch_types"]
                
            if effect == "shift":
                slices = self.create_slices(height, width, params)
                for direction, start, end in slices:
                    offset = int(params["chaos_amount"] * 50)
                    if direction == "h":
                        result[start:end] = np.roll(result[start:end], offset, axis=1)
                    else:
                        result[:, start:end] = np.roll(result[:, start:end], offset, axis=0)
                        
            elif effect == "repeat":
                slice_size = np.random.randint(2, 10)
                if np.random.random() < 0.5:
                    y = np.random.randint(0, height - slice_size)
                    result[y:y+slice_size] = np.tile(result[y:y+1], (slice_size, 1, 1))
                else:
                    x = np.random.randint(0, width - slice_size)
                    result[:, x:x+slice_size] = np.tile(result[:, x:x+1], (1, slice_size, 1))
                    
            elif effect == "mirror":
                if np.random.random() < 0.5:
                    y = np.random.randint(0, height//2)
                    size = np.random.randint(10, 50)
                    result[y:y+size] = np.flip(result[y:y+size], axis=1)
                else:
                    x = np.random.randint(0, width//2)
                    size = np.random.randint(10, 50)
                    result[:, x:x+size] = np.flip(result[:, x:x+size], axis=0)
                    
            elif effect == "noise":
                noise_mask = np.random.random(image.shape[:2]) < params["chaos_amount"] * 0.1
                noise = np.random.random((noise_mask.sum(), 3))
                result[noise_mask] = noise
                
        return result

    def apply_distortions(self, image, params):
        """Apply various distortion effects"""
        result = image.copy()
        
        if params["wave_distortion"] > 0:
            height, width = image.shape[:2]
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            
            wave = np.sin(x/30) * params["wave_distortion"] * 20
            x_displaced = x + wave
            
            for c in range(3):
                result[..., c] = ndimage.map_coordinates(image[..., c], 
                                                       [y, x_displaced], 
                                                       order=1, 
                                                       mode='reflect')
                
        if params["compression_artifacts"] > 0:
            quality = int((1 - params["compression_artifacts"]) * 90 + 10)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', (result * 255).astype(np.uint8), encode_param)
            decoded = cv2.imdecode(encoded, 1).astype(np.float32) / 255
            result = decoded[..., ::-1]  # BGR to RGB
            
        if params["pixel_sorting"] > 0:
            brightness = np.mean(result, axis=2)
            threshold = np.percentile(brightness, params["pixel_sorting"] * 100)
            
            for i in range(result.shape[0]):
                mask = brightness[i] > threshold
                if np.any(mask):
                    row = result[i, mask]
                    sorted_indices = np.argsort(np.mean(row, axis=1))
                    result[i, mask] = row[sorted_indices]
                
        return result

    def process_single_image(self, image, params):
        """Process a single image with all effects"""
        result = image.copy()
        
        result = self.apply_glitch_patterns(result, params)
        result = self.apply_color_shift(result, params)
        result = self.apply_distortions(result, params)
        
        if params["preserve_bright_areas"] > 0:
            brightness = np.max(image, axis=2)
            mask = brightness > params["preserve_bright_areas"]
            result[mask] = image[mask]
            
        return np.clip(result, 0, 1)

    def generate_databend(self, images, slice_direction, slice_min_size, slice_max_size,
                         slice_variability, channel_shift_mode, color_intensity, 
                         rgb_shift_separate, preserve_bright_areas, glitch_types,
                         pattern_frequency, chaos_amount, seed, wave_distortion,
                         compression_artifacts, pixel_sorting, control_after_generate):
        
        if seed != -1:
            np.random.seed(seed)
            
        device = images.device
        batch_size = images.shape[0]
        
        params = locals()
        del params['self']
        del params['images']
        del params['device']
        del params['batch_size']
        del params['control_after_generate']
        
        output_batch = []
        
        print(f"\n{'='*50}")
        print(f"Applying DataBend effects:")
        print(f"Batch size: {batch_size}")
        
        with tqdm(total=batch_size, desc="Processing images") as pbar:
            for b in range(batch_size):
                img = images[b].cpu().numpy()
                canvas = self.process_single_image(img, params)
                canvas_tensor = torch.from_numpy(canvas).float()
                output_batch.append(canvas_tensor)
                pbar.update(1)
        
        result = torch.stack(output_batch).to(device)
        
        print(f"\nProcessing complete!")
        print(f"{'='*50}")
        
        return (result,)