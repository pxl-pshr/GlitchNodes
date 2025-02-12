# https://x.com/_pxlpshr

import torch
import torch.nn.functional as F
from tqdm import tqdm
import time

class interferenceV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "horizontal_iterations": ("INT", {
                    "default": 10, 
                    "min": 0, 
                    "max": 50,
                    "step": 1
                }),
                "vertical_iterations": ("INT", {
                    "default": 4, 
                    "min": 0, 
                    "max": 50,
                    "step": 1
                }),
                "shift_amount": ("INT", {
                    "default": -1, 
                    "min": -10, 
                    "max": 10,
                    "step": 1
                }),
                "color_shift": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "color_mode": (["monochrome", "rainbow", "duotone", "invert"],),
                "preserve_brightness": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_sort_shader"
    CATEGORY = "image/postprocessing"

    def apply_sort_shader(self, image, horizontal_iterations, vertical_iterations, 
                         shift_amount, color_shift, color_mode, preserve_brightness):
        start_time = time.time()
        
        image = image.float()
        B, H, W, C = image.shape

        def hash(x):
            return torch.frac(torch.sin(x * 12.9898 + x * 78.233) * 43758.5453)

        def cv(c):
            return c.sum(dim=-1)
        
        def apply_color_effect(img, color_val):
            if color_mode == "monochrome":
                return img
            
            elif color_mode == "rainbow":
                # Create rainbow effect based on pixel position and hash
                hue = (hash(color_val) * color_shift).unsqueeze(-1).expand(-1, -1, -1, C)
                if preserve_brightness:
                    brightness = img.mean(dim=-1, keepdim=True)
                    return (img + hue) * brightness
                return img + hue
            
            elif color_mode == "duotone":
                # Create a two-color effect
                color1 = torch.tensor([0.8, 0.2, 0.2]).to(img.device)  # Reddish
                color2 = torch.tensor([0.2, 0.2, 0.8]).to(img.device)  # Bluish
                mask = (hash(color_val) > 0.5).unsqueeze(-1)
                return torch.where(mask, img * color1, img * color2)
            
            elif color_mode == "invert":
                # Selectively invert colors based on hash value
                mask = (hash(color_val) > 0.5).unsqueeze(-1)
                return torch.where(mask, 1.0 - img, img)
            
            return img

        def compare(c1, c2, p, i):
            condition = (p[..., 0] % 2) != (i % 2)
            cv1 = cv(c1)
            cv2 = cv(c2)
            
            result = torch.where(
                cv1.unsqueeze(-1) > cv2.unsqueeze(-1),
                c2,
                torch.where(condition.unsqueeze(-1), 
                          hash(cv1).unsqueeze(-1).expand(-1, -1, -1, C),
                          torch.where(cv1.unsqueeze(-1) > cv2.unsqueeze(-1), c1, c2))
            )
            
            return apply_color_effect(result, cv1)

        def compare_h(c1, c2, p, i):
            condition = (p[..., 1] % 2) != (i % 2)
            cv1 = cv(c1)
            cv2 = cv(c2)
            
            result = torch.where(
                cv1.unsqueeze(-1) > cv2.unsqueeze(-1),
                c2,
                torch.where(condition.unsqueeze(-1), 
                          hash(cv1).unsqueeze(-1).expand(-1, -1, -1, C),
                          torch.where(cv1.unsqueeze(-1) > cv2.unsqueeze(-1), c1, c2))
            )
            
            return apply_color_effect(result, cv1)

        pos = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=-1).to(image.device)
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1)

        def sort_horizontal(image, iterations):
            h_start = time.time()
            for i in tqdm(range(iterations), desc="Horizontal Sort"):
                image = compare(image, torch.roll(image, shifts=shift_amount, dims=2), pos, i)
            h_time = time.time() - h_start
            print(f"Horizontal sort: {h_time:.3f}s")
            return image

        def sort_vertical(image, iterations):
            v_start = time.time()
            for i in tqdm(range(iterations), desc="Vertical Sort"):
                image = compare_h(image, torch.roll(image, shifts=shift_amount, dims=1), pos, i)
            v_time = time.time() - v_start
            print(f"Vertical sort: {v_time:.3f}s")
            return image

        # Apply sorting with progress bars
        if horizontal_iterations > 0:
            image = sort_horizontal(image, horizontal_iterations)
        if vertical_iterations > 0:
            image = sort_vertical(image, vertical_iterations)

        total_time = time.time() - start_time
        print(f"Total time: {total_time:.3f}s")

        return (image,)