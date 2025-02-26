# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
import cv2
from tqdm import tqdm

class LuminousFlow:
    """
    A ComfyUI node that transforms images into flowing luminous strands,
    creating an ethereal effect of light threads that follow the image's features.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "line_spacing": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 50,
                    "step": 1
                }),
                "line_thickness": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1
                }),
                "flow_intensity": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                }),
                "smoothing": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "glow_intensity": ("FLOAT", {
                    "default": 12.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1
                }),
                "darkness": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "vibrancy": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.1,
                    "max": 15.0,
                    "step": 0.1
                }),
                "glow_spread": ("INT", {
                    "default": 7,
                    "min": 0,
                    "max": 12,
                    "step": 1
                }),
                "contrast": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_luminous_flow"
    CATEGORY = "GlitchNodes"

    def enhance_colors(self, color, vibrancy, contrast):
        """Enhanced color processing with neon effect"""
        # Convert to HSV for better color manipulation
        color_rgb = np.clip(color * 255, 0, 255).astype(np.uint8).reshape(1, 1, 3)
        color_hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)
        
        # Super aggressive saturation enhancement
        color_hsv[0, 0, 1] = np.clip(color_hsv[0, 0, 1] * vibrancy * 1.5, 0, 255)
        
        # Enhanced value/brightness with extra pop
        color_hsv[0, 0, 2] = np.clip(color_hsv[0, 0, 2] * contrast * 1.5, 0, 255)
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)
        enhanced_color = enhanced_rgb[0, 0] / 255.0
        
        # Additional contrast enhancement with more aggressive gamma
        enhanced_color = np.power(enhanced_color, 0.7)
        enhanced_color = np.power(enhanced_color, 1/contrast)
        
        # Extra boost to bright areas
        enhanced_color = np.where(enhanced_color > 0.5, 
                                enhanced_color * 1.3,
                                enhanced_color * 0.7)
        
        return np.clip(enhanced_color, 0, 1)

    def draw_line(self, img, start_point, end_point, color, thickness=1, glow_spread=0):
        """Enhanced glow effect with neon quality"""
        if glow_spread > 0:
            # Create super intense central glow
            center_glow = color * 2.5
            cv2.line(img, 
                    (int(start_point[0]), int(start_point[1])), 
                    (int(end_point[0]), int(end_point[1])), 
                    np.clip(center_glow, 0, 1).tolist(),
                    thickness + 2,
                    cv2.LINE_AA)
            
            # Create multiple layers of glow with increased intensity
            for i in range(glow_spread, 0, -1):
                glow_color = color * (2.5 / (i + 0.2))
                cv2.line(img, 
                        (int(start_point[0]), int(start_point[1])), 
                        (int(end_point[0]), int(end_point[1])), 
                        np.clip(glow_color, 0, 1).tolist(),
                        thickness + i*2,
                        cv2.LINE_AA)
            
            # Add an extra intense inner glow
            inner_glow = color * 3.0
            cv2.line(img, 
                    (int(start_point[0]), int(start_point[1])), 
                    (int(end_point[0]), int(end_point[1])), 
                    np.clip(inner_glow, 0, 1).tolist(),
                    thickness + 1,
                    cv2.LINE_AA)
        
        # Draw main line with increased intensity
        cv2.line(img, 
                (int(start_point[0]), int(start_point[1])), 
                (int(end_point[0]), int(end_point[1])), 
                np.clip(color * 1.5, 0, 1).tolist(),
                thickness,
                cv2.LINE_AA)

    def process_image(self, img, params, batch_idx=0, total_batches=1):
        height, width = img.shape[:2]
        
        print(f"\nProcessing image {batch_idx + 1}/{total_batches}")
        print(f"Image size: {width}x{height}")
        
        # Enhanced preprocessing with more contrast
        intensity_map = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        intensity_map = cv2.GaussianBlur(intensity_map, (3, 3), 0)
        
        # More aggressive contrast in intensity map
        intensity_map = np.power(intensity_map, params["contrast"] * 1.2)
        intensity_map = (intensity_map - intensity_map.min()) / (intensity_map.max() - intensity_map.min() + 1e-7)
        
        # Create darker background
        background = img * params["darkness"] * 0.8
        canvas = background.copy()
        
        max_lines = (height - 2 * params["line_spacing"]) // params["line_spacing"]
        line_positions = np.linspace(params["line_spacing"], height - params["line_spacing"], max_lines)
        
        print(f"Generating {len(line_positions)} luminous lines...")
        
        with tqdm(total=len(line_positions), desc="Drawing lines", unit="line") as pbar:
            for pos in line_positions:
                points = []
                colors = []
                
                for x in range(width):
                    y_pos = int(pos)
                    # Enhanced color processing
                    base_color = img[min(y_pos, height-1), x]
                    color = base_color * params["glow_intensity"]
                    color = self.enhance_colors(color, params["vibrancy"], params["contrast"])
                    color = np.clip(color, 0, 1)
                    colors.append(color)
                    
                    displacement = -intensity_map[y_pos, x] * params["line_spacing"] * params["flow_intensity"]
                    new_y = np.clip(pos + displacement, 0, height - 1)
                    points.append((float(x), float(new_y)))
                
                if params["smoothing"] > 0:
                    points = np.array(points)
                    points[:, 1] = gaussian_filter1d(points[:, 1], params["smoothing"])
                
                for i in range(len(points) - 1):
                    self.draw_line(canvas, 
                                 points[i], 
                                 points[i + 1], 
                                 colors[i], 
                                 params["line_thickness"],
                                 params["glow_spread"])
                
                pbar.update(1)
        
        return canvas

    def create_luminous_flow(self, image, line_spacing, line_thickness, flow_intensity,
                           smoothing, glow_intensity, darkness, vibrancy, glow_spread,
                           contrast):
        batch_size = image.shape[0]
        
        print(f"\n{'='*50}")
        print("Starting Luminous Flow generation")
        print(f"Batch size: {batch_size}")
        print(f"Parameters:")
        print(f"- Line spacing: {line_spacing}")
        print(f"- Flow intensity: {flow_intensity}")
        print(f"- Glow intensity: {glow_intensity}")
        print(f"- Vibrancy: {vibrancy}")
        print(f"- Contrast: {contrast}")
        print(f"- Glow spread: {glow_spread}")
        print(f"{'='*50}\n")
        
        image_np = image.cpu().numpy()
        
        params = {
            "line_spacing": line_spacing,
            "line_thickness": line_thickness,
            "flow_intensity": flow_intensity,
            "smoothing": smoothing,
            "glow_intensity": glow_intensity,
            "darkness": darkness,
            "vibrancy": vibrancy,
            "glow_spread": glow_spread,
            "contrast": contrast
        }
        
        output_batch = []
        with tqdm(total=batch_size, desc="Processing batch", unit="image") as pbar:
            for i in range(batch_size):
                result = self.process_image(image_np[i], params, i, batch_size)
                result_tensor = torch.from_numpy(result).float()
                output_batch.append(result_tensor)
                pbar.update(1)
        
        print(f"\n{'='*50}")
        print("Processing complete!")
        print(f"{'='*50}")
        
        return (torch.stack(output_batch),)