# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/
# Original repo: https://github.com/collidingScopes/ascii

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import torch
from tqdm import tqdm

class ASCII:
    CATEGORY = 'GlitchNodes'
    FUNCTION = 'execute'
    OUTPUT_NODE = False
    RETURN_TYPES = ('IMAGE',)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'IMAGE': ('IMAGE',),
                'background': ('STRING', {'default': '#080c37'}),
                'bgGradient': ('BOOLEAN', {'default': False}),
                'bgSaturation': ('INT', {'default': 60, 'min': 0, 'max': 100}),
                'fontColor': ('STRING', {'default': '#c7205b'}),
                'fontColor2': ('STRING', {'default': '#00ff61'}),
                'fontSizeFactor': ('FLOAT', {'default': 3.0, 'min': 0.1, 'max': 10.0}),
                'resolution': ('INT', {'default': 137, 'min': 1}),
                'threshold': ('INT', {'default': 0, 'min': 0, 'max': 255}),
                'invert': ('BOOLEAN', {'default': True}),
                'randomness': ('INT', {'default': 15, 'min': 0, 'max': 100}),
                'textType': (['Random Text', 'Input Text'], {}),
                'textInput': ('STRING', {'default': 'pxlpshr', 'multiline': False}),
            }
        }

    def execute(
        self, IMAGE, background, bgGradient, bgSaturation,
        fontColor, fontColor2, fontSizeFactor, resolution,
        threshold, invert, randomness, textType, textInput
    ):
        # Convert input to PIL images
        pil_images = self._tensor_to_pil(IMAGE)

        # Generate ASCII versions with progress bar
        ascii_images = []
        for img in tqdm(pil_images, desc='Generating ASCII'):
            ascii_images.append(
                self._make_ascii(
                    img, background, fontColor, fontColor2,
                    fontSizeFactor, resolution, threshold,
                    invert, randomness, textType, textInput
                )
            )

        # Convert each PIL ASCII image to a torch tensor (H,W,C) float32 in [0,1]
        ascii_tensors = []
        for img in tqdm(ascii_images, desc='Converting to tensors'):
            arr = np.array(img).astype(np.float32) / 255.0
            # ensure HWC
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            tensor = torch.from_numpy(arr)
            ascii_tensors.append(tensor)

        # Return list of tensors for ComfyUI save_images
        return (ascii_tensors,)

    def _tensor_to_pil(self, image):
        # Handle dict input
        if isinstance(image, dict):
            image = image.get('samples', image)
        # Already PIL images
        if isinstance(image, list) and all(isinstance(i, Image.Image) for i in image):
            return image
        # Convert tensors or arrays to numpy
        try:
            if hasattr(image, 'cpu'):
                arr = image.cpu().numpy()
            else:
                arr = np.array(image)
        except:
            arr = np.array(image)
        pil_list = []
        # Batch: B,C,H,W or B,H,W,C
        if arr.ndim == 4:
            for t in arr:
                pil_list.append(self._array_to_pil(t))
        else:
            pil_list.append(self._array_to_pil(arr))
        return pil_list

    def _array_to_pil(self, arr):
        # Move channel-first to HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        # Scale floats
        if issubclass(arr.dtype.type, np.floating):
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        # Grayscale to RGB
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return Image.fromarray(arr)

    def _make_ascii(
        self, pil_img, background, fc1, fc2, fsf,
        res, thr, inv, rnd, ttype, tinput
    ):
        w, h = pil_img.size
        # Create background
        result = Image.new('RGB', (w, h), background)
        draw = ImageDraw.Draw(result)
        # Load font
        try:
            font = ImageFont.truetype('DejaVuSansMono.ttf', int(fsf * 10))
        except:
            font = ImageFont.load_default()
        # Compute grid size
        cell_w = w / res
        try:
            mask = font.getmask('A')
            glyph_w, glyph_h = mask.size
        except AttributeError:
            bbox = font.getbbox('A')
            glyph_w = bbox[2] - bbox[0]
            glyph_h = bbox[3] - bbox[1]
        cell_h = cell_w * (glyph_h / glyph_w)
        rows = max(1, int(h / cell_h))
        # Downsample and get luminance
        small = pil_img.resize((res, rows))
        L = np.array(small.convert('L'))
        # Choose character set
        if ttype == 'Input Text' and tinput:
            chars = list(tinput)
        else:
            chars = list('@%#*+=-:. ')
        # Draw ASCII
        for i in range(rows):
            for j in range(res):
                lum = int(L[i, j])
                if thr > 0:
                    cond = lum > thr
                    if inv:
                        cond = not cond
                    if not cond:
                        continue
                if rnd > 0 and random.random() < rnd / 100:
                    ch = random.choice(chars)
                else:
                    idx = int(lum / 255 * (len(chars) - 1))
                    ch = chars[idx]
                color = fc1 if ((i + j) % 2 == 0) else fc2
                draw.text((j * cell_w, i * cell_h), ch, fill=color, font=font)
        return result