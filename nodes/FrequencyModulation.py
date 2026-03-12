# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import torch
import torch.nn.functional as F
from torch import nn
import comfy.utils
import logging

logger = logging.getLogger(__name__)

class FrequencyModulation:
    """Applies frequency modulation effects to create glitchy visual artifacts."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "carrier_frequency": ("FLOAT", {"default": 10, "min": 0.01, "max": 10.0, "step": 0.01}),
                "bandwidth": ("FLOAT", {"default": 10, "min": 0.1, "max": 10.0, "step": 0.1}),
                "quantization": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "colorspace": (["RGB", "YUV"],),
                "first_channel_only": ("BOOLEAN", {"default": False}),
                "lowpass1_on": ("BOOLEAN", {"default": True}),
                "lowpass2_on": ("BOOLEAN", {"default": True}),
                "lowpass3_on": ("BOOLEAN", {"default": True}),
                "lowpass1_cutoff": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01}),
                "lowpass2_cutoff": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "lowpass3_cutoff": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0, "step": 0.01}),
                "negate": ("BOOLEAN", {"default": False}),
                "blend_mode": (["NONE", "ADD", "SUBTRACT", "MULTIPLY", "SCREEN", "OVERLAY"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_fm"
    CATEGORY = "GlitchNodes"
    DESCRIPTION = "Modulates image using frequency domain techniques with customizable carrier frequency and bandwidth"

    def apply_fm(self, image, carrier_frequency, bandwidth, quantization, colorspace, first_channel_only,
                 lowpass1_on, lowpass2_on, lowpass3_on, lowpass1_cutoff, lowpass2_cutoff, lowpass3_cutoff,
                 negate, blend_mode):
        logger.info("Starting frequency modulation processing...")
        try:
            pbar = comfy.utils.ProgressBar(7)
            # ComfyUI IMAGE is [B,H,W,C] — convert to [B,C,H,W] for conv2d processing
            img = image.float().permute(0, 3, 1, 2)
            original_img = img.clone()
            pbar.update(1)

            img = self.to_colorspace(img, colorspace)
            if first_channel_only:
                img = img[:, 0:1]
            pbar.update(1)

            b, c, h, w = img.shape
            phase = torch.cumsum(img, dim=-1) * bandwidth
            t = torch.arange(w, dtype=torch.float32, device=img.device).view(1, 1, 1, -1).repeat(b, c, h, 1)
            modulated = torch.cos(2 * torch.pi * carrier_frequency * t + phase)
            pbar.update(1)

            if quantization > 0:
                modulated = torch.round(modulated * quantization) / quantization
            pbar.update(1)

            demodulated = torch.diff(modulated, dim=-1)
            demodulated = F.pad(demodulated, (1, 0, 0, 0), mode='replicate')
            pbar.update(1)

            if lowpass1_on:
                demodulated = self.apply_lowpass(demodulated, lowpass1_cutoff)
            if lowpass2_on:
                demodulated = self.apply_lowpass(demodulated, lowpass2_cutoff)
            if lowpass3_on:
                demodulated = self.apply_lowpass(demodulated, lowpass3_cutoff)
            pbar.update(1)

            denom = demodulated.max() - demodulated.min()
            if denom < 1e-8:
                result = torch.zeros_like(demodulated)
            else:
                result = (demodulated - demodulated.min()) / denom
            if negate:
                result = 1 - result

            result = self.from_colorspace(result, colorspace)
            if blend_mode != "NONE":
                result = self.apply_blend_mode(original_img, result, blend_mode)
            pbar.update(1)

            # Convert back from [B,C,H,W] to [B,H,W,C]
            result = result.permute(0, 2, 3, 1)

            logger.info("Frequency modulation processing completed")
            return (result.float().clamp(0, 1),)
        except Exception as e:
            logger.error(f"Error in frequency modulation processing: {str(e)}")
            raise

    def apply_lowpass(self, signal, cutoff):
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device=signal.device) / 16
        kernel = kernel.view(1, 1, 3, 3).repeat(signal.shape[1], 1, 1, 1)
        return F.conv2d(signal, kernel, padding=1, groups=signal.shape[1])

    def apply_blend_mode(self, img1, img2, mode):
        if mode == "ADD":
            return torch.clamp(img1 + img2, 0, 1)
        elif mode == "SUBTRACT":
            return torch.clamp(img1 - img2, 0, 1)
        elif mode == "MULTIPLY":
            return img1 * img2
        elif mode == "SCREEN":
            return 1 - (1 - img1) * (1 - img2)
        elif mode == "OVERLAY":
            return torch.where(img1 < 0.5, 2 * img1 * img2, 1 - 2 * (1 - img1) * (1 - img2))
        return img2

    def to_colorspace(self, img, colorspace):
        if colorspace == "RGB":
            return img
        elif colorspace == "YUV":
            return self.rgb_to_yuv(img)
        return img

    def from_colorspace(self, img, colorspace):
        if colorspace == "RGB":
            return img
        elif colorspace == "YUV":
            return self.yuv_to_rgb(img)
        return img

    def rgb_to_yuv(self, rgb):
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v = 0.615 * r - 0.51499 * g - 0.10001 * b
        return torch.cat([y, u, v], dim=1)

    def yuv_to_rgb(self, yuv):
        y, u, v = yuv[:, 0:1], yuv[:, 1:2], yuv[:, 2:3]
        r = y + 1.13983 * v
        g = y - 0.39465 * u - 0.58060 * v
        b = y + 2.03211 * u
        return torch.cat([r, g, b], dim=1)