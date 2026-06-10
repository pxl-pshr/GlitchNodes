# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/
# Original Repo: https://github.com/collidingScopes/ascii

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import torch
import comfy.utils
import logging

logger = logging.getLogger(__name__)

class ASCII:
    """Converts images to ASCII art with customizable font colors and characters."""
    CATEGORY = 'GlitchNodes'
    FUNCTION = 'execute'
    OUTPUT_NODE = False
    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)
    DESCRIPTION = "Renders images as ASCII art with gradient text coloring and optional background effects"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'IMAGE': ('IMAGE',),
                'background': ('STRING', {'default': '#080c37'}),
                'fontColor': ('STRING', {'default': '#c7205b'}),
                'fontColor2': ('STRING', {'default': '#00ff61'}),
                'fontSizeFactor': ('FLOAT', {'default': 3.0, 'min': 0.1, 'max': 10.0}),
                'resolution': ('INT', {'default': 137, 'min': 1, 'max': 512}),
                'threshold': ('INT', {'default': 0, 'min': 0, 'max': 255}),
                'invert': ('BOOLEAN', {'default': True}),
                'randomness': ('INT', {'default': 15, 'min': 0, 'max': 100}),
                'textType': (['Random Text', 'Input Text'], {}),
                'textInput': ('STRING', {'default': 'pxlpshr', 'multiline': False}),
                'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
            }
        }

    def execute(
        self, IMAGE, background,
        fontColor, fontColor2, fontSizeFactor, resolution,
        threshold, invert, randomness, textType, textInput, seed
    ):
        # 1) Convert input to PIL
        pil_images = self._tensor_to_pil(IMAGE)

        # 2) Make ASCII
        ascii_images = []
        pbar_ascii = comfy.utils.ProgressBar(len(pil_images))
        for frame_index, img in enumerate(pil_images):
            rng = random.Random(seed + frame_index)
            ascii_images.append(
                self._make_ascii(
                    img, background, fontColor, fontColor2,
                    fontSizeFactor, resolution, threshold,
                    invert, randomness, textType, textInput, rng
                )
            )
            pbar_ascii.update(1)

        # 3) Back to tensors
        ascii_tensors = []
        for img in ascii_images:
            arr = np.array(img).astype(np.float32) / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            ascii_tensors.append(torch.from_numpy(arr))

        result = torch.stack(ascii_tensors).float().clamp(0, 1)
        if hasattr(IMAGE, 'device'):
            result = result.to(IMAGE.device)
        return (result,)

    def _tensor_to_pil(self, image):
        if isinstance(image, dict):
            image = image.get('samples', image)
        if isinstance(image, list) and all(isinstance(i, Image.Image) for i in image):
            return image
        try:
            arr = image.cpu().numpy() if hasattr(image, 'cpu') else np.array(image)
        except Exception:
            arr = np.array(image)
        out = []
        if arr.ndim == 4:
            for t in arr:
                out.append(self._array_to_pil(t))
        else:
            out.append(self._array_to_pil(arr))
        return out

    def _array_to_pil(self, arr):
        # Only treat as CHW if the last dim is not a plausible channel count
        if arr.ndim == 3 and arr.shape[-1] not in (1, 3, 4) and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if issubclass(arr.dtype.type, np.floating):
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        return Image.fromarray(arr)

    def _parse_hex_color(self, value, fallback):
        s = str(value).strip().lstrip('#')
        if len(s) == 3:
            s = ''.join(c * 2 for c in s)
        if len(s) == 6:
            try:
                return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))
            except ValueError:
                pass
        logger.warning(f"Invalid hex color '{value}', using fallback {fallback}")
        return fallback

    def _make_ascii(
        self, pil_img, background, fc1, fc2, fsf,
        res, thr, inv, rnd, ttype, tinput, rng
    ):
        w, h = pil_img.size
        cell_w = w / res

        # parse colors with safe fallbacks
        bg = self._parse_hex_color(background, (8, 12, 55))
        c1 = self._parse_hex_color(fc1, (199, 32, 91))
        c2 = self._parse_hex_color(fc2, (0, 255, 97))

        canvas = Image.new('RGB', (w, h), bg)

        # scale Pillow's built-in font to the cell size (Pillow >= 10.1)
        try:
            font = ImageFont.load_default(size=max(1.0, cell_w * fsf))
        except TypeError:
            font = ImageFont.load_default()

        # measure one character to set grid
        mask = font.getmask('A')
        glyph_w, glyph_h = mask.size
        if glyph_w == 0 or glyph_h == 0:
            glyph_w, glyph_h = 6, 11
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

        # pre-render a glyph atlas: one mask per unique character
        atlas = {}
        for ch in set(chars):
            bbox = font.getbbox(ch)
            gw, gh = max(0, int(bbox[2])), max(0, int(bbox[3]))
            if gw == 0 or gh == 0:
                atlas[ch] = None
                continue
            tile = Image.new('L', (gw, gh), 0)
            ImageDraw.Draw(tile).text((0, 0), ch, fill=255, font=font)
            atlas[ch] = tile

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
                if rnd > 0 and rng.random() < rnd/100:
                    ch = rng.choice(chars)
                else:
                    idx = int(lum/255*(len(chars)-1))
                    ch = chars[idx]

                glyph = atlas.get(ch)
                if glyph is None:
                    continue

                # blend color
                t = lum / 255.0
                color = (
                    int(c1[0]*(1-t) + c2[0]*t),
                    int(c1[1]*(1-t) + c2[1]*t),
                    int(c1[2]*(1-t) + c2[2]*t),
                )

                canvas.paste(color, (int(j*cell_w), int(i*cell_h)), glyph)

        return canvas
