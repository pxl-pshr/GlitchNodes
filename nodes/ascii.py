# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/
# Original Repo: https://github.com/collidingScopes/ascii

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
        # 1) Convert input to PIL
        pil_images = self._tensor_to_pil(IMAGE)

        # 2) Make ASCII
        ascii_images = []
        for img in tqdm(pil_images, desc='Generating ASCII'):
            ascii_images.append(
                self._make_ascii(
                    img, background, fontColor, fontColor2,
                    fontSizeFactor, resolution, threshold,
                    invert, randomness, textType, textInput
                )
            )

        # 3) Back to tensors
        ascii_tensors = []
        for img in tqdm(ascii_images, desc='Converting to tensors'):
            arr = np.array(img).astype(np.float32) / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            ascii_tensors.append(torch.from_numpy(arr))

        return (ascii_tensors,)

    def _tensor_to_pil(self, image):
        if isinstance(image, dict):
            image = image.get('samples', image)
        if isinstance(image, list) and all(isinstance(i, Image.Image) for i in image):
            return image
        try:
            arr = image.cpu().numpy() if hasattr(image, 'cpu') else np.array(image)
        except:
            arr = np.array(image)
        out = []
        if arr.ndim == 4:
            for t in arr:
                out.append(self._array_to_pil(t))
        else:
            out.append(self._array_to_pil(arr))
        return out

    def _array_to_pil(self, arr):
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if issubclass(arr.dtype.type, np.floating):
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        return Image.fromarray(arr)

    def _make_ascii(
        self, pil_img, background, fc1, fc2, fsf,
        res, thr, inv, rnd, ttype, tinput
    ):
        w, h = pil_img.size
        canvas = Image.new('RGB', (w, h), background)
        draw = ImageDraw.Draw(canvas)

        # -- use Pillow's built-in bitmap font --
        font = ImageFont.load_default()

        # measure one character to set grid
        mask = font.getmask('A')
        glyph_w, glyph_h = mask.size
        cell_w = w / res
        cell_h = cell_w * (glyph_h / glyph_w)
        rows = max(1, int(h / cell_h))

        # get luminance map
        small = pil_img.resize((res, rows))
        L = np.array(small.convert('L'), dtype=np.uint8)

        # reversed charset: space first → '@' last
        if ttype == 'Input Text' and tinput:
            chars = list(tinput)
        else:
            chars = list(' .:-=+*#%@')

        # parse colors
        c1 = tuple(int(fc1.lstrip('#')[i:i+2], 16) for i in (0,2,4))
        c2 = tuple(int(fc2.lstrip('#')[i:i+2], 16) for i in (0,2,4))

        for i in range(rows):
            for j in range(res):
                lum = int(L[i, j])
                if inv:
                    lum = 255 - lum
                if lum == 0:
                    continue
                if thr > 0 and lum < thr:
                    continue

                # pick glyph
                if rnd > 0 and random.random() < rnd/100:
                    ch = random.choice(chars)
                else:
                    idx = int(lum/255*(len(chars)-1))
                    ch = chars[idx]

                # blend color
                t = lum / 255.0
                color = (
                    int(c1[0]*(1-t) + c2[0]*t),
                    int(c1[1]*(1-t) + c2[1]*t),
                    int(c1[2]*(1-t) + c2[2]*t),
                )

                draw.text((j*cell_w, i*cell_h), ch, fill=color, font=font)

        return canvas
