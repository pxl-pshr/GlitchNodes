# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import logging
import torch
import comfy.utils

logger = logging.getLogger(__name__)

class Interference:
    """Apply bayer-matrix-inspired sorting and color effects."""
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
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_sort_shader"
    CATEGORY = "GlitchNodes"
    DESCRIPTION = "Apply bayer-matrix-inspired pixel sorting with various color effects"

    def apply_sort_shader(self, image, horizontal_iterations, vertical_iterations,
                         shift_amount, color_shift, color_mode, preserve_brightness):
        device = image.device
        image = image.float()
        B, H, W, C = image.shape

        # Normalize channels: expand grayscale, split off alpha
        alpha = None
        if C == 1:
            image = image.expand(-1, -1, -1, 3).contiguous()
        elif C == 4:
            alpha = image[..., 3:4]
            image = image[..., :3].contiguous()
        else:
            image = image[..., :3]
        C = image.shape[-1]

        # Duotone palette constants
        duo_color1 = torch.tensor([0.8, 0.2, 0.2], device=device)  # Reddish
        duo_color2 = torch.tensor([0.2, 0.2, 0.8], device=device)  # Bluish

        def hash(x):
            return torch.frac(torch.sin(x * 12.9898 + torch.roll(x, 1, dims=-1) * 78.233) * 43758.5453)

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
                mask = (hash(color_val) > 0.5).unsqueeze(-1)
                return torch.where(mask, img * duo_color1, img * duo_color2)

            elif color_mode == "invert":
                # Selectively invert colors based on hash value
                mask = (hash(color_val) > 0.5).unsqueeze(-1)
                return torch.where(mask, 1.0 - img, img)

            return img

        def compare(c1, c2, p, i):
            # horizontal pass sorts along width — parity follows column index
            condition = ((p[..., 1] % 2) != (i % 2)).unsqueeze(-1)
            cv1 = cv(c1).unsqueeze(-1)
            cv2 = cv(c2).unsqueeze(-1)
            return torch.where(condition & (cv1 > cv2), c2, c1)

        def compare_h(c1, c2, p, i):
            # vertical pass sorts along height — parity follows row index
            condition = ((p[..., 0] % 2) != (i % 2)).unsqueeze(-1)
            cv1 = cv(c1).unsqueeze(-1)
            cv2 = cv(c2).unsqueeze(-1)
            return torch.where(condition & (cv1 > cv2), c2, c1)

        pos = torch.stack(torch.meshgrid(torch.arange(H, device=device),
                                         torch.arange(W, device=device),
                                         indexing="ij"), dim=-1)
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1)

        # Create single progress bar for both operations
        total_iterations = horizontal_iterations + vertical_iterations
        pbar = comfy.utils.ProgressBar(total_iterations)

        def sort_horizontal(image, iterations):
            for i in range(iterations):
                image = compare(image, torch.roll(image, shifts=shift_amount, dims=2), pos, i)
                pbar.update(1)
            return image

        def sort_vertical(image, iterations):
            for i in range(iterations):
                image = compare_h(image, torch.roll(image, shifts=shift_amount, dims=1), pos, i)
                pbar.update(1)
            return image

        # Apply sorting with progress bars
        if horizontal_iterations > 0:
            image = sort_horizontal(image, horizontal_iterations)
        if vertical_iterations > 0:
            image = sort_vertical(image, vertical_iterations)

        # Apply the color effect once after sorting
        image = apply_color_effect(image, cv(image))
        image = image.clamp(0, 1)

        if alpha is not None:
            image = torch.cat([image, alpha], dim=-1)

        return (image.float().to(device),)