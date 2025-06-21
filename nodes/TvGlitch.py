# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LowpassFilter:
    def __init__(self):
        self.timeInterval = 0.0
        self.cutoff = 0.0
        self.alpha = 0.0
        self.prev = 0.0
        self.tau = 0.0

    def setFilter(self, rate, hz):
        self.timeInterval = 1.0 / rate
        self.tau = 1.0 / (hz * 2 * np.pi)
        self.cutoff = hz
        self.alpha = self.timeInterval / (self.tau + self.timeInterval)

    def resetFilter(self, val=0):
        self.prev = val

    def lowpass(self, sample):
        stage1 = sample * self.alpha
        stage2 = self.prev - (self.prev * self.alpha)
        self.prev = stage1 + stage2
        return self.prev

    def highpass(self, sample):
        return sample - self.lowpass(sample)


class TvGlitch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "subcarrier_amplitude": ("INT", {"default": 40, "min": 1, "max": 200, "step": 1}),
                "video_noise": ("INT", {"default": 100, "min": 0, "max": 10000, "step": 100}),
                "video_chroma_noise": ("INT", {"default": 100, "min": 0, "max": 10000, "step": 100}),
                "video_chroma_phase_noise": ("INT", {"default": 15, "min": 0, "max": 100, "step": 1}),
                "video_chroma_loss": ("FLOAT", {"default": 0.24, "min": 0.0, "max": 1.0, "step": 0.01}),
                "composite_preemphasis": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "scanlines_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 5.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_tv_glitch"
    CATEGORY = "GlitchNodes"

    def apply_tv_glitch(
        self,
        image,
        subcarrier_amplitude,
        video_noise,
        video_chroma_noise,
        video_chroma_phase_noise,
        video_chroma_loss,
        composite_preemphasis,
        scanlines_scale,
    ):
        logger.info("Starting TV glitch effect processing...")

        try:
            image = image.float()
            if image.max() > 1.0:
                image = image / 255.0

            processed_images = []
            for i in tqdm(range(image.shape[0]), desc="Processing frames"):
                img_np = image[i].cpu().numpy()
                img_cv = self.process_frame(
                    img_np,
                    subcarrier_amplitude,
                    video_noise,
                    video_chroma_noise,
                    video_chroma_phase_noise,
                    video_chroma_loss,
                    composite_preemphasis,
                )

                if scanlines_scale > 1:
                    img_cv = self.render_scanlines(img_cv, scanlines_scale)

                processed_images.append(img_cv)

            img_np = np.stack(processed_images, axis=0)
            img_tensor = torch.from_numpy(img_np).to(image.device)
            img_tensor = (img_tensor * 255).byte()

            logger.info("TV glitch effect completed")
            return (img_tensor,)

        except Exception as e:
            logger.error(f"Error in TV glitch processing: {e!s}")
            raise

    def process_frame(
        self,
        src,
        subcarrier_amplitude,
        video_noise,
        video_chroma_noise,
        video_chroma_phase_noise,
        video_chroma_loss,
        composite_preemphasis,
    ):
        h, w, _ = src.shape
        dst = np.zeros_like(src)

        fY, fI, fQ = self.rgb_to_yiq(src)
        self.composite_lowpass(fI, fQ, w, h)
        self.chroma_into_luma(fY, fI, fQ, w, h, subcarrier_amplitude)

        if composite_preemphasis != 0.0:
            pre = LowpassFilter()
            pre.setFilter((315000000.0 * 4.0) / 88.0, 315000000 / 88.0)
            for y in range(h):
                pre.resetFilter(16)
                for x in range(w):
                    s = fY[y, x]
                    s += pre.highpass(s) * composite_preemphasis
                    fY[y, x] = s

        if video_noise != 0:
            noise = 0
            noise_mod = video_noise * 2 + 1
            for i in range(h * w):
                fY[i // w, i % w] += noise
                noise += np.random.randint(noise_mod) - video_noise
                noise = noise / 2

        self.chroma_from_luma(fY, fI, fQ, w, h, subcarrier_amplitude)

        if video_chroma_noise != 0:
            noiseI, noiseQ = 0, 0
            noise_mod = video_chroma_noise * 2 + 1
            for i in range(h * w):
                fI[i // w, i % w] += noiseI
                fQ[i // w, i % w] += noiseQ
                noiseI += np.random.randint(noise_mod) - video_chroma_noise
                noiseI = noiseI / 2
                noiseQ += np.random.randint(noise_mod) - video_chroma_noise
                noiseQ = noiseQ / 2

        if video_chroma_phase_noise != 0:
            noise = 0
            noise_mod = (video_chroma_phase_noise * 2) + 1
            for y in range(h):
                noise += np.random.randint(noise_mod) - video_chroma_phase_noise
                noise = noise / 2
                pi = (noise * np.pi) / 100.0
                sinpi, cospi = np.sin(pi), np.cos(pi)
                for x in range(w):
                    u, v = fI[y, x], fQ[y, x]
                    fI[y, x] = (u * cospi) - (v * sinpi)
                    fQ[y, x] = (u * sinpi) + (v * cospi)

        if video_chroma_loss != 0:
            for y in range(h):
                if np.random.rand() < video_chroma_loss:
                    fI[y, :] = 0
                    fQ[y, :] = 0

        return self.yiq_to_rgb(fY, fI, fQ)

    def rgb_to_yiq(self, rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        i = 0.596 * r - 0.274 * g - 0.322 * b
        q = 0.211 * r - 0.523 * g + 0.312 * b
        return y * 256.0, i * 256.0, q * 256.0

    def yiq_to_rgb(self, y, i, q):
        y, i, q = y / 256.0, i / 256.0, q / 256.0
        r = y + 0.956 * i + 0.621 * q
        g = y - 0.272 * i - 0.647 * q
        b = y - 1.106 * i + 1.703 * q
        return np.clip(np.stack([r, g, b], axis=-1), 0, 1)

    def composite_lowpass(self, fI, fQ, w, h):
        for p in range(2):
            P = fI if p == 0 else fQ
            cutoff = 1300000.0 if p == 0 else 600000.0
            delay = 2 if p == 0 else 4

            lp = [LowpassFilter() for _ in range(3)]
            for f in range(3):
                lp[f].setFilter((315000000.0 * 4.0) / 88.0, cutoff)

            for y in range(h):
                for f in range(3):
                    lp[f].resetFilter()

                for x in range(w):
                    s = P[y, x]
                    for f in range(3):
                        s = lp[f].lowpass(s)
                    if x >= delay:
                        P[y, x - delay] = s

    def chroma_into_luma(self, fY, fI, fQ, w, h, subcarrier_amplitude):
        Umult = [1, 0, -1, 0]
        Vmult = [0, 1, 0, -1]

        for y in range(h):
            xi = 0
            for x in range(w):
                sxi = (x + xi) & 3
                chroma = fI[y, x] * subcarrier_amplitude * Umult[sxi]
                chroma += fQ[y, x] * subcarrier_amplitude * Vmult[sxi]
                fY[y, x] += chroma / 50
                fI[y, x] = fQ[y, x] = 0

    def chroma_from_luma(self, fY, fI, fQ, w, h, subcarrier_amplitude):
        for y in range(h):
            chroma = np.zeros(w)
            delay = [0, 0, 0, 0]
            delay[2] = fY[y, 0]
            delay[3] = fY[y, 1]
            sum_val = delay[2] + delay[3]

            for x in range(w):
                c = fY[y, x + 2] if x + 2 < w else 0

                sum_val -= delay[0]
                for j in range(3):
                    delay[j] = delay[j + 1]
                delay[3] = c
                sum_val += delay[3]
                fY[y, x] = sum_val / 4
                chroma[x] = c - fY[y, x]

            xi = 0
            for x in range((4 - xi) & 3, w - 3, 4):
                chroma[x + 2] = -chroma[x + 2]
                chroma[x + 3] = -chroma[x + 3]

            chroma = (chroma * 50) / subcarrier_amplitude

            for x in range(0, w - xi - 1, 2):
                fI[y, x] = -chroma[x + xi]
                fQ[y, x] = -chroma[x + xi + 1]

            for x in range(0, w - 2, 2):
                fI[y, x + 1] = (fI[y, x] + fI[y, x + 2]) / 2
                fQ[y, x + 1] = (fQ[y, x] + fQ[y, x + 2]) / 2

    def render_scanlines(self, img, scale):
        h, w, _ = img.shape
        scanline_img = np.zeros_like(img)

        for y in range(0, h, 3):
            scanline_img[y, :, 0] = np.random.uniform(0.8, 0.9)
            if y + 1 < h:
                scanline_img[y + 1, :, 1] = np.random.uniform(0.8, 0.9)
            if y + 2 < h:
                scanline_img[y + 2, :, 2] = np.random.uniform(0.8, 0.9)

        blurred = F.interpolate(
            torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0),
            scale_factor=1 / scale,
            mode="bilinear",
            align_corners=False,
        )
        blurred = F.interpolate(blurred, size=(h, w), mode="bilinear", align_corners=False)
        blurred = blurred.squeeze().permute(1, 2, 0).numpy()

        result = img * 0.7 + blurred * 0.3 + scanline_img * 0.15
        return np.clip(result, 0, 1)
