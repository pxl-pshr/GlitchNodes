import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

class Peaks:
    DIRECTIONS = ["horizontal", "vertical"]
    COLOR_MODES = ["monochrome", "rainbow"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "spacing": ("INT", {
                    "default": 10,
                    "min": 2,
                    "max": 50,
                    "step": 1
                }),
                "background_color": ("COLOR", {"default": "#000000"}),
                "line_color": ("COLOR", {"default": "#FFFFFF"}),
                "intensity": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                }),
                "line_thickness": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1
                }),
                "smoothing": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "direction": (cls.DIRECTIONS,),
                "color_mode": (cls.COLOR_MODES,)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_peaks"
    CATEGORY = "image/processing"

    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB array"""
        hex_color = hex_color.lstrip('#')
        return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0

    def get_rainbow_color(self, progress):
        h = progress
        s = 0.8
        v = 0.8
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        if h < 1/6:
            rgb = (c, x, 0)
        elif h < 2/6:
            rgb = (x, c, 0)
        elif h < 3/6:
            rgb = (0, c, x)
        elif h < 4/6:
            rgb = (0, x, c)
        elif h < 5/6:
            rgb = (x, 0, c)
        else:
            rgb = (c, 0, x)
        return np.array([r + m for r in rgb], dtype=np.float32)

    def draw_antialiased_point(self, canvas, x, y, color):
        """Draw a single anti-aliased point"""
        x_floor, x_frac = divmod(x, 1)
        y_floor, y_frac = divmod(y, 1)
        x_floor = int(x_floor)
        y_floor = int(y_floor)
        
        if 0 <= x_floor < canvas.shape[1]-1 and 0 <= y_floor < canvas.shape[0]-1:
            for dx, dy in [(0,0), (0,1), (1,0), (1,1)]:
                nx = x_floor + dx
                ny = y_floor + dy
                alpha = (1 - abs(x_frac - dx)) * (1 - abs(y_frac - dy))
                canvas[ny, nx] = canvas[ny, nx] * (1 - alpha) + color * alpha

    def draw_line(self, canvas, x0, y0, x1, y1, color, thickness):
        """Draw an anti-aliased line with uniform thickness"""
        dx = x1 - x0
        dy = y1 - y0
        distance = max(abs(dx), abs(dy))
        
        if distance == 0:
            self.draw_antialiased_point(canvas, x0, y0, color)
            return
            
        half_thick = thickness / 2
        
        for i in range(int(distance) + 1):
            t = i / distance
            x = x0 + t * dx
            y = y0 + t * dy
            
            # Draw perpendicular line for thickness
            if abs(dx) > abs(dy):
                for j in range(-int(half_thick), int(half_thick) + 1):
                    self.draw_antialiased_point(canvas, x, y + j, color)
            else:
                for j in range(-int(half_thick), int(half_thick) + 1):
                    self.draw_antialiased_point(canvas, x + j, y, color)

    def process_single_image(self, img, height, width, channels, params, batch_idx=0, total_batches=1, pbar=None):
        self.current_direction = params["direction"]
        
        # Convert to displacement map
        if channels > 1:
            displacement_map = (0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2])
        else:
            displacement_map = img[..., 0].copy()
            
        # Normalize displacement map
        displacement_map = (displacement_map - displacement_map.min()) / (displacement_map.max() - displacement_map.min() + 1e-7)
        
        # Initialize canvas
        canvas = np.full((height, width, 3), self.hex_to_rgb(params["background_color"]), dtype=np.float32)
        
        # Create uniform array of line positions
        if params["direction"] == "vertical":
            displacement_map = displacement_map.T
            max_lines = (width - 2 * params["spacing"]) // params["spacing"]
            line_positions = np.linspace(params["spacing"], width - params["spacing"], max_lines)
        else:
            max_lines = (height - 2 * params["spacing"]) // params["spacing"]
            line_positions = np.linspace(params["spacing"], height - params["spacing"], max_lines)
        
        # Pre-compute line colors
        if params["color_mode"] == "monochrome":
            line_color = self.hex_to_rgb(params["line_color"])
            colors = np.array([line_color] * len(line_positions))
        else:
            colors = np.array([self.get_rainbow_color(i / len(line_positions)) for i in range(len(line_positions))])
        
        # Draw lines
        for idx, (pos, color) in enumerate(zip(line_positions, colors)):
            if pbar:
                pbar.set_description(f"Image {batch_idx+1}/{total_batches} - Line {idx+1}/{len(line_positions)}")
                pbar.update(0)
            
            # Calculate points
            points = []
            if params["direction"] == "vertical":
                for y in range(height):
                    displacement = -displacement_map[int(pos), y] * params["spacing"] * params["intensity"]
                    new_x = np.clip(pos + displacement, 0, width - 1)
                    points.append((float(new_x), float(y)))
            else:
                for x in range(width):
                    displacement = -displacement_map[int(pos), x] * params["spacing"] * params["intensity"]
                    new_y = np.clip(pos + displacement, 0, height - 1)
                    points.append((float(x), float(new_y)))
            
            # Apply smoothing if needed
            if params["smoothing"] > 0:
                points = np.array(points)
                if params["direction"] == "vertical":
                    points[:, 0] = gaussian_filter1d(points[:, 0], params["smoothing"])
                else:
                    points[:, 1] = gaussian_filter1d(points[:, 1], params["smoothing"])
            
            # Draw line segments
            for i in range(len(points) - 1):
                x0, y0 = points[i]
                x1, y1 = points[i + 1]
                self.draw_line(canvas, x0, y0, x1, y1, color, params["line_thickness"])
            
            if pbar:
                pbar.update(1)
        
        return canvas

    def generate_peaks(self, images, spacing, background_color, line_color, intensity,
                      line_thickness, smoothing, direction, color_mode):
        device = images.device
        batch_size, height, width, channels = images.shape
        
        print(f"\n{'='*50}")
        print(f"Starting peaks generation:")
        print(f"Batch size: {batch_size}")
        print(f"Image dimensions: {height}x{width}")
        print(f"Parameters:")
        print(f"- Direction: {direction}")
        print(f"- Spacing: {spacing}")
        print(f"- Line thickness: {line_thickness}")
        print(f"- Smoothing: {smoothing}")
        print(f"- Color mode: {color_mode}")
        print(f"{'='*50}")
        
        params = {
            "spacing": spacing,
            "background_color": background_color,
            "line_color": line_color,
            "intensity": intensity,
            "line_thickness": line_thickness,
            "smoothing": smoothing,
            "direction": direction,
            "color_mode": color_mode
        }
        
        output_batch = []
        
        with tqdm(total=batch_size, desc="Processing images", unit="image") as pbar:
            for b in range(batch_size):
                img = images[b].cpu().numpy()
                canvas = self.process_single_image(
                    img, height, width, channels, params,
                    batch_idx=b, total_batches=batch_size,
                    pbar=pbar
                )
                canvas_tensor = torch.from_numpy(canvas).float()
                output_batch.append(canvas_tensor)
                pbar.update(1)
        
        result = torch.stack(output_batch).to(device)
        
        print(f"\n{'='*50}")
        print(f"Processing complete!")
        print(f"Final output shape: {result.shape}")
        print(f"{'='*50}")
        
        return (result,)