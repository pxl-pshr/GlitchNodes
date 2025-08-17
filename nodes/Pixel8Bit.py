# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

from typing import List
import torch
import numpy as np
from PIL import Image
import re

# --- tqdm (with safe fallback) ---
try:
    from tqdm import tqdm as _tqdm
    def tqdm(iterable, **kwargs):
        return _tqdm(iterable, **kwargs)
except Exception:
    def tqdm(iterable, **kwargs):
        # no-op progress if tqdm missing
        return iterable

########################
# Utility: Palettes
########################

def _hex_to_rgb01(hex_str: str) -> np.ndarray:
    hex_str = hex_str.strip().lstrip('#')
    if len(hex_str) == 3:
        hex_str = ''.join([c*2 for c in hex_str])
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return np.array([r, g, b], dtype=np.float32) / 255.0

def _palette_from_hex_list(hex_list: List[str]) -> np.ndarray:
    cols = []
    for h in hex_list:
        h = h.strip()
        if not h:
            continue
        cols.append(_hex_to_rgb01(h))
    if not cols:
        # fallback to simple grayscale 4
        cols = [np.array([v, v, v], dtype=np.float32) for v in (0.0, 0.33, 0.66, 1.0)]
    return np.stack(cols, axis=0)  # (K,3)

def _palette_vga_256() -> np.ndarray:
    """
    Standard 256-color VGA palette approximation:
    - 6x6x6 RGB cube with levels {0, 51, 102, 153, 204, 255} (216 colors)
    - 40-step grayscale ramp commonly used in VGA (approximation)
    To keep things compact and flexible, we generate it on the fly.
    """
    levels = np.array([0, 51, 102, 153, 204, 255], dtype=np.float32)
    cube = np.stack(np.meshgrid(levels, levels, levels, indexing='ij'), axis=-1).reshape(-1, 3)
    # Grayscale ramp: 0..255 spaced into 40 steps (skip 0 & 255 duplication with cube extremes if desired)
    gray_steps = np.linspace(0, 255, 40, dtype=np.float32)[:, None]
    grays = np.repeat(gray_steps, 3, axis=1)
    pal = np.vstack([cube, grays])
    pal01 = pal / 255.0
    # Deduplicate any exact duplicates just in case
    pal01 = np.unique((pal01 * 255).round().astype(np.uint8), axis=0).astype(np.float32) / 255.0
    return pal01

# Curated retro palettes (sRGB approximations widely used by emulators / pixel tools)
PALETTES = {
    # Existing sets
    "PICO-8": [
        "#000000","#1D2B53","#7E2553","#008751","#AB5236","#5F574F","#C2C3C7","#FFF1E8",
        "#FF004D","#FFA300","#FFEC27","#00E436","#29ADFF","#83769C","#FF77A8","#FFCCAA"
    ],
    "GameBoy": [  # DMG-01 4 shades
        "#0F380F","#306230","#8BAC0F","#9BBC0F"
    ],
    "NES": [  # Common NES subset (from NTSC approximations / emulator standards)
        "#000000","#545454","#A8A8A8","#FCFCFC","#001E74","#081090","#4E4E4E","#D8D8D8",
        "#0078F8","#3CBCFC","#B8B8F8","#F8F8F8","#00B800","#58D854","#D8F878","#B8F8B8",
        "#F87858","#F8B878","#F8D8B8"
    ],
    "C64": [  # VIC-II 16 colors
        "#000000","#FFFFFF","#883932","#67B6BD","#8B3F96","#55A049","#40318D","#BFCE72",
        "#8B5429","#574200","#B86962","#505050","#808080","#94E089","#7869C4","#9F9F9F"
    ],

    # New sets
    "ZX Spectrum": [
        # 7 base colors + bright versions + black
        "#000000",  # black
        "#0000D7","#D70000","#D700D7","#00D700","#00D7D7","#D7D700","#D7D7D7",  # normal
        "#0000FF","#FF0000","#FF00FF","#00FF00","#00FFFF","#FFFF00","#FFFFFF"   # bright
    ],
    "Apple II": [
        # Composite artifact colors (approx) - 6 common choices for pixel art workflows
        "#000000","#DD0000","#00DD00","#0000DD","#DDDD00","#DDDDDD"
    ],
    "EGA-16": [  # IBM EGA 16 color palette
        "#000000","#0000AA","#00AA00","#00AAAA",
        "#AA0000","#AA00AA","#AA5500","#AAAAAA",
        "#555555","#5555FF","#55FF55","#55FFFF",
        "#FF5555","#FF55FF","#FFFF55","#FFFFFF"
    ],
    "VGA-256": "DYNAMIC_VGA_256",  # generated in code
    "Amiga-Workbench": [  # Popular Workbench/demoscene-leaning set (subset; Amiga was 12-bit 4096 color capable)
        "#000000","#FFFFFF","#880000","#AAFFEE","#CC44CC","#00CC55","#0000AA","#EEEE77",
        "#DD8855","#664400","#FF7777","#333333","#777777","#AAFF66","#0088FF","#BBBBBB",
        "#444444","#008888","#888800","#880088","#888888","#004488","#448800","#884400"
    ],
    "Atari2600-Subset": [  # a commonly used 16-color subset of Stella NTSC approximations
        "#000000","#1A1A1A","#2F2F2F","#555555","#7B7B7B","#A0A0A0","#C6C6C6","#EDEDED",
        "#6A1E0C","#CC2E0B","#FF6E27","#FF9E4A","#0B5E8A","#1A9AC9","#3FD0FF","#72ECFF"
    ],
    "MSX": [  # TMS9918-ish palette used in MSX1 / SG-1000 era
        "#000000","#3EB849","#74D07D","#5955E0","#8076F1","#B95E51","#65DBEF","#DB6559",
        "#FF897D","#CCC35E","#DEDEDE","#AAFF66","#0088FF","#BBBBBB","#FFFFFF"
    ],
    # Optional bonus: CGA (old PC 4-color mode palettes)
    "CGA-Default": [
        "#000000","#55FFFF","#FF55FF","#FFFFFF"  # black + cyan + magenta + white
    ],
    "CGA-Alternate": [
        "#000000","#55FF55","#FF5555","#FFFF55"  # black + green + red + yellow
    ],
}

def _get_fixed_palette(name: str, custom_hex: str) -> np.ndarray:
    if name == "Custom":
        hexes = re.split(r'[,\s]+', custom_hex.strip())
        return _palette_from_hex_list([h for h in hexes if h])

    if name == "VGA-256":
        return _palette_vga_256()

    if name not in PALETTES:
        # fallback to PICO-8 if unknown
        return _palette_from_hex_list(PALETTES["PICO-8"])
    entry = PALETTES[name]
    if isinstance(entry, str) and entry == "DYNAMIC_VGA_256":
        return _palette_vga_256()
    return _palette_from_hex_list(entry)

########################
# Utility: Dither matrices
########################

def _bayer_matrix(n: int) -> np.ndarray:
    if n == 2:
        M = np.array([[0, 2],
                      [3, 1]], dtype=np.float32)
    elif n == 4:
        M = np.array([[ 0,  8,  2, 10],
                      [12,  4, 14,  6],
                      [ 3, 11,  1,  9],
                      [15,  7, 13,  5]], dtype=np.float32)
    elif n == 8:
        # construct 8 via recursive pattern
        M2 = _bayer_matrix(4)
        M = np.block([
            [4*M2 + 0,  4*M2 + 2],
            [4*M2 + 3,  4*M2 + 1]
        ]).astype(np.float32)
    else:
        raise ValueError("Ordered dither size must be 2, 4, or 8")
    return (M + 0.5) / (M.size)  # normalize 0..1

########################
# Quantizers
########################

def _nearest_palette(img: np.ndarray, palette: np.ndarray) -> np.ndarray:
    # img: (H,W,3) float in [0,1]; palette: (K,3)
    H, W, _ = img.shape
    flat = img.reshape(-1, 3)
    diffs = flat[:, None, :] - palette[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=2)
    idx = np.argmin(dist2, axis=1)
    quant = palette[idx]
    return quant.reshape(H, W, 3)

def _kmeans_palette(img: np.ndarray, k: int, iters: int = 10, seed: int = 42) -> np.ndarray:
    H, W, _ = img.shape
    flat = img.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    if flat.shape[0] > 20000:
        sample_idx = rng.choice(flat.shape[0], size=20000, replace=False)
        sample = flat[sample_idx]
    else:
        sample = flat

    c_idx = rng.choice(sample.shape[0], size=max(2, k), replace=False)
    centers = sample[c_idx].copy()

    for _ in range(iters):
        d = sample[:, None, :] - centers[None, :, :]
        dist2 = np.sum(d*d, axis=2)
        labels = np.argmin(dist2, axis=1)
        for j in range(k):
            pts = sample[labels == j]
            if pts.size > 0:
                centers[j] = pts.mean(axis=0)

    # sort by luminance for nicer ordering
    lum = (0.2126*centers[:,0] + 0.7152*centers[:,1] + 0.0722*centers[:,2])
    order = np.argsort(lum)
    return centers[order]

def _posterize(img: np.ndarray, bits: int) -> np.ndarray:
    if bits >= 8:
        return img
    levels = (1 << bits) - 1
    return np.round(np.clip(img, 0, 1) * levels) / levels

########################
# Dithering
########################

def _ordered_dither(img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    H, W, C = img.shape
    n = matrix.shape[0]
    tiled = np.tile(matrix, (H // n + 1, W // n + 1))[:H, :W]
    thresh = tiled - 0.5
    out = img + thresh[..., None] / max(8, n*n)  # subtle nudge
    return np.clip(out, 0.0, 1.0)

def _floyd_steinberg(img: np.ndarray, palette: np.ndarray) -> np.ndarray:
    H, W, _ = img.shape
    work = img.copy()
    out = np.zeros_like(work)
    for y in range(H):
        for x in range(W):
            old = work[y, x]
            new = _nearest_palette(old[None, None, :], palette)[0, 0]
            out[y, x] = new
            err = old - new
            if x+1 < W:
                work[y, x+1] = np.clip(work[y, x+1] + err * (7/16), 0, 1)
            if y+1 < H and x > 0:
                work[y+1, x-1] = np.clip(work[y+1, x-1] + err * (3/16), 0, 1)
            if y+1 < H:
                work[y+1, x] = np.clip(work[y+1, x] + err * (5/16), 0, 1)
            if y+1 < H and x+1 < W:
                work[y+1, x+1] = np.clip(work[y+1, x+1] + err * (1/16), 0, 1)
    return out

########################
# Core transform
########################

def _ensure_rgb01(arr: np.ndarray) -> np.ndarray:
    if arr.shape[-1] == 3:
        return arr
    if arr.shape[-1] == 4:
        return arr[..., :3]
    if arr.shape[-1] == 1:
        return np.repeat(arr, 3, axis=-1)
    # unexpected channels
    return arr[..., :3]

def _to_pil_uint8(img01: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(img01 * 255.0, 0, 255).astype(np.uint8), mode="RGB")

def _from_pil_uint8(pil: Image.Image) -> np.ndarray:
    arr = np.asarray(pil).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr

def _pixelate(img01: np.ndarray, pixel_size: int) -> np.ndarray:
    if pixel_size <= 1:
        return img01
    H, W, _ = img01.shape
    down_w = max(1, W // pixel_size)
    down_h = max(1, H // pixel_size)
    pil = _to_pil_uint8(img01)
    small = pil.resize((down_w, down_h), resample=Image.NEAREST)
    back = small.resize((W, H), resample=Image.NEAREST)
    return _from_pil_uint8(back)

def _apply_gamma(img01: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        return img01
    return np.power(np.clip(img01, 0, 1), 1.0/gamma)

def _quantize(img01: np.ndarray, mode: str, palette_name: str, custom_hex: str, k_colors: int,
              dithering: str, ordered_size: int, posterize_bits: int) -> np.ndarray:
    if 1 <= posterize_bits < 8:
        img01 = _posterize(img01, posterize_bits)

    if mode == "Fixed Palette":
        pal = _get_fixed_palette(palette_name, custom_hex)
    else:
        k = max(2, int(k_colors))
        pal = _kmeans_palette(img01, k)

    if dithering == "Floyd-Steinberg":
        return _floyd_steinberg(img01, pal)
    elif dithering == "Ordered":
        M = _bayer_matrix(int(ordered_size))
        pre = _ordered_dither(img01, M)
        return _nearest_palette(pre, pal)
    else:
        return _nearest_palette(img01, pal)

def _process_frame(frame01: np.ndarray,
                   pixel_size: int,
                   mode: str,
                   palette_name: str,
                   custom_hex: str,
                   k_colors: int,
                   dithering: str,
                   ordered_size: int,
                   posterize_bits: int,
                   gamma: float) -> np.ndarray:
    img = _ensure_rgb01(frame01)
    if gamma != 1.0:
        img = _apply_gamma(img, gamma)
    img = _pixelate(img, pixel_size)
    img = _quantize(img, mode, palette_name, custom_hex, k_colors, dithering, ordered_size, posterize_bits)
    return np.clip(img, 0.0, 1.0)

########################
# ComfyUI Node
########################

class Pixel8Bit:
    """
    Convert image(s) to an 8-bit / retro look:
    - Pixelation
    - Palette quantization (Fixed or Adaptive K-Means)
    - Dithering (Ordered / Floyd–Steinberg)
    - Rich fixed palettes: PICO-8, Game Boy, NES, C64, ZX Spectrum, Apple II, EGA-16, VGA-256 (dynamic),
                           Amiga Workbench, Atari 2600 subset, MSX, CGA variants
    Shows per-frame progress in terminal via tqdm (safe fallback if missing).
    """
    CATEGORY = "GlitchNodes"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "pixel_size": ("INT", {"default": 4, "min": 1, "max": 256, "step": 1}),
                "quant_mode": (["Fixed Palette", "Adaptive K-Means"],),
                "fixed_palette": ([
                    "PICO-8", "GameBoy", "NES", "C64",
                    "ZX Spectrum", "Apple II", "EGA-16", "VGA-256",
                    "Amiga-Workbench", "Atari2600-Subset", "MSX",
                    "CGA-Default", "CGA-Alternate", "Custom"
                ],),
                "custom_palette_hex": ("STRING", {
                    "default": "#000000,#555555,#AAAAAA,#FFFFFF",
                    "multiline": False
                }),
                "k_colors": ("INT", {"default": 8, "min": 2, "max": 256, "step": 1}),
                "dithering": (["None", "Ordered", "Floyd-Steinberg"],),
                "ordered_size": ("INT", {"default": 4, "min": 2, "max": 8, "step": 2}),
                "posterize_bits": ("INT", {"default": 8, "min": 1, "max": 8, "step": 1}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05}),
            }
        }

    def execute(self,
                IMAGE: torch.Tensor,
                pixel_size: int,
                quant_mode: str,
                fixed_palette: str,
                custom_palette_hex: str,
                k_colors: int,
                dithering: str,
                ordered_size: int,
                posterize_bits: int,
                gamma: float):

        # Expect IMAGE in [B,H,W,C] float32 0..1
        if not isinstance(IMAGE, torch.Tensor):
            raise ValueError("IMAGE must be a torch.Tensor")
        img = IMAGE
        if img.dim() != 4:
            raise ValueError(f"Expected IMAGE tensor of shape [B,H,W,C], got {tuple(img.shape)}")

        B, H, W, C = img.shape

        # Move to CPU for numpy ops (keeps VRAM usage low)
        img_np = img.detach().cpu().numpy().astype(np.float32)
        out_frames = []

        # Progress bar in terminal
        for b in tqdm(range(B), desc="[Pixel8Bit] Processing frames", unit="frame"):
            frame = img_np[b]
            if C not in (1, 3, 4):
                frame = frame[..., :3]
            frame = _ensure_rgb01(frame)

            out = _process_frame(
                frame,
                pixel_size=pixel_size,
                mode=quant_mode,
                palette_name=fixed_palette,
                custom_hex=custom_palette_hex,
                k_colors=k_colors,
                dithering=dithering,
                ordered_size=ordered_size,
                posterize_bits=posterize_bits,
                gamma=gamma
            )
            out_frames.append(out)

        out_np = np.stack(out_frames, axis=0)  # [B,H,W,3]
        out_t = torch.from_numpy(out_np).to(IMAGE.device, dtype=IMAGE.dtype)
        return (out_t,)