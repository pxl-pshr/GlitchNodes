# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import torch
import numpy as np
from PIL import Image
import io
import random
from tqdm import tqdm

def tensor_to_bytes(image):
    return (image.cpu().numpy() * 255).astype(np.uint8)

class GlitchIT:
    def __init__(self):
        self.SOS = b"\xFF\xDA"  # Start Of Scan
        self.EOI = b"\xFF\xD9"  # End Of Image

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": -2**63, "max": 2**63 - 1}),
                "min_amount": ("INT", {"default": 1, "min": 0, "max": 100}),
                "max_amount": ("INT", {"default": 10, "min": 1, "max": 100}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_glitch"
    CATEGORY = "GlitchNodes"

    def apply_glitch(self, images, seed, min_amount, max_amount):
        # Validate and fix the range
        if min_amount > max_amount:
            min_amount, max_amount = max_amount, min_amount

        original_shape = images.shape
        glitched_images = []

        for image in tqdm(images, desc="Applying glitch effect"):
            np_image = tensor_to_bytes(image)
            
            if np_image.shape == (1, 1, 576):
                working_image = np_image.squeeze()
            elif np_image.shape[-1] == 3:
                working_image = np_image
            else:
                working_image = np.stack([np_image] * 3, axis=-1)
            
            pil_image = Image.fromarray(working_image.squeeze(), mode='RGB')
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=95)
            original = buffer.getvalue()

            prng = random.Random(seed)
            amount = prng.randint(min_amount, max_amount)
            start = original.index(self.SOS) + len(self.SOS) + 10
            end = original.rindex(self.EOI)
            data = bytearray(original[start:end])

            glitched = set()
            for _ in range(amount):
                while True:
                    index = prng.randrange(len(data))
                    if index not in glitched and data[index] not in [0, 255]:
                        glitched.add(index)
                        break
                while True:
                    value = prng.randint(1, 254)
                    if data[index] != value:
                        data[index] = value
                        break

            glitched_jpeg = original[:start] + data + original[end:]
            glitched_image = np.array(Image.open(io.BytesIO(glitched_jpeg)))

            if np_image.shape == (1, 1, 576):
                glitched_image = glitched_image.mean(axis=2, keepdims=True).reshape(1, 1, 576)
            elif np_image.shape[0] == 1:
                glitched_image = glitched_image.mean(axis=2, keepdims=True)
            else:
                glitched_image = glitched_image.transpose(2, 0, 1)
            
            glitched_images.append(glitched_image)

        result = torch.from_numpy(np.stack(glitched_images).astype(np.float32) / 255.0)
        
        if original_shape != result.shape:
            result = result.permute(0, 2, 3, 1)

        return (result,)