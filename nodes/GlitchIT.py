# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import torch
import numpy as np
import logging
from PIL import Image
import io
import random
import comfy.utils

logger = logging.getLogger(__name__)

def tensor_to_bytes(image):
    return (image.cpu().numpy() * 255).astype(np.uint8)

class GlitchIT:
    """Apply JPEG glitch effects by manipulating JPEG scan data"""
    def __init__(self):
        self.SOS = b"\xFF\xDA"  # Start Of Scan
        self.EOI = b"\xFF\xD9"  # End Of Image

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "min_amount": ("INT", {"default": 1, "min": 0, "max": 100}),
                "max_amount": ("INT", {"default": 10, "min": 1, "max": 100}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_glitch"
    CATEGORY = "GlitchNodes"
    DESCRIPTION = "Apply JPEG glitch effects by manipulating JPEG scan data bytes"

    def apply_glitch(self, images, seed, min_amount, max_amount):
        try:
            # Validate and fix the range
            if min_amount > max_amount:
                min_amount, max_amount = max_amount, min_amount

            device = images.device
            glitched_images = []
            pbar = comfy.utils.ProgressBar(len(images))

            for frame_index, image in enumerate(images):
                np_image = tensor_to_bytes(image)

                if np_image.ndim == 3 and np_image.shape[-1] >= 3:
                    working_image = np_image[..., :3]
                elif np_image.ndim == 3 and np_image.shape[-1] == 1:
                    working_image = np.repeat(np_image, 3, axis=-1)
                else:
                    working_image = np.stack([np_image] * 3, axis=-1)

                pil_image = Image.fromarray(working_image, mode='RGB')
                buffer = io.BytesIO()
                pil_image.save(buffer, format="JPEG", quality=95)
                original = buffer.getvalue()

                prng = random.Random(seed + frame_index)
                amount = prng.randint(min_amount, max_amount)
                start = original.index(self.SOS) + len(self.SOS) + 10
                end = original.rindex(self.EOI)
                start = min(start, end)
                data = bytearray(original[start:end])

                eligible = [i for i, byte in enumerate(data) if byte not in (0, 255)]
                for index in prng.sample(eligible, min(amount, len(eligible))):
                    value = prng.randint(1, 254)
                    if value == data[index]:
                        value = value + 1 if value < 254 else 1
                    data[index] = value

                glitched_jpeg = original[:start] + bytes(data) + original[end:]
                glitched_image = np.array(Image.open(io.BytesIO(glitched_jpeg)).convert('RGB'))  # (H,W,3)

                glitched_images.append(glitched_image)
                pbar.update(1)

            result = torch.from_numpy(np.stack(glitched_images).astype(np.float32) / 255.0)
            result = result.clamp(0, 1)

            if images.shape[-1] == 4:
                alpha = images[..., 3:4].detach().float().cpu()
                result = torch.cat([result, alpha], dim=-1)

            return (result.to(device),)
        except Exception as e:
            logger.error(f"Error in GlitchIT processing: {str(e)}")
            raise