# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import torch
import torch.nn.functional as F
import numpy as np
import logging
import comfy.utils

logger = logging.getLogger(__name__)


def _iir_lowpass_1d(signal, alpha):
    """First-order IIR lowpass: y[n] = alpha*x[n] + (1-alpha)*y[n-1].

    Uses vectorized numpy approach — processes entire 1D signal at once
    by leveraging the geometric series expansion of the IIR filter.
    For short filter memory (alpha not too small), this is accurate and fast.
    """
    out = np.empty_like(signal)
    if len(signal) == 0:
        return out
    out[0] = signal[0] * alpha
    beta = 1.0 - alpha
    for i in range(1, len(signal)):
        out[i] = alpha * signal[i] + beta * out[i - 1]
    return out


def _iir_lowpass_rows(arr, alpha, delay=0, passes=1):
    """Apply first-order IIR lowpass per row with optional delay shift.

    Uses scipy.signal.lfilter when available, falls back to row-level loop.
    """
    b = np.array([alpha], dtype=np.float64)
    a = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)

    try:
        from scipy.signal import lfilter
        result = arr.copy()
        for _ in range(passes):
            result = lfilter(b, a, result, axis=-1)
    except ImportError:
        # Fallback: per-row loop (still much faster than per-pixel Python)
        result = arr.copy()
        beta = 1.0 - alpha
        for _ in range(passes):
            for y in range(result.shape[0]):
                row = result[y]
                for i in range(1, len(row)):
                    row[i] = alpha * row[i] + beta * row[i - 1]

    if delay > 0:
        result = np.roll(result, -delay, axis=-1)
        result[:, -delay:] = 0
    return result


def _iir_noise_signal(deltas):
    """Compute IIR random walk: noise[n] = 0.5 * noise[n-1] + delta[n].

    Uses scipy.signal.lfilter when available, falls back to cumulative loop.
    """
    try:
        from scipy.signal import lfilter
        return lfilter([1.0], [1.0, -0.5], deltas)
    except ImportError:
        out = np.empty_like(deltas)
        out[0] = deltas[0]
        for i in range(1, len(deltas)):
            out[i] = 0.5 * out[i - 1] + deltas[i]
        return out


class TvGlitch:
    """Applies analog TV glitch effects with color distortion and scanlines."""
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
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_tv_glitch"
    CATEGORY = "GlitchNodes"
    DESCRIPTION = "Simulates analog TV glitch effects including chroma noise, video noise, and scanlines"

    def apply_tv_glitch(self, image, subcarrier_amplitude, video_noise, video_chroma_noise,
                       video_chroma_phase_noise, video_chroma_loss, composite_preemphasis, scanlines_scale):
        logger.info("Starting TV glitch effect processing...")

        try:
            image = image.float()
            if image.max() > 1.0:
                image = image / 255.0

            pbar = comfy.utils.ProgressBar(image.shape[0])
            processed_images = []
            for i in range(image.shape[0]):
                img_np = image[i].cpu().numpy()
                img_cv = self.process_frame(img_np, subcarrier_amplitude, video_noise, video_chroma_noise,
                                         video_chroma_phase_noise, video_chroma_loss, composite_preemphasis)

                if scanlines_scale > 1:
                    img_cv = self.render_scanlines(img_cv, scanlines_scale)

                processed_images.append(img_cv)
                pbar.update(1)

            img_np = np.stack(processed_images, axis=0)
            img_tensor = torch.from_numpy(img_np).to(image.device)
            img_tensor = img_tensor.float().clamp(0, 1)

            logger.info("TV glitch effect completed")
            return (img_tensor,)

        except Exception as e:
            logger.error(f"Error in TV glitch processing: {str(e)}")
            raise

    def process_frame(self, src, subcarrier_amplitude, video_noise, video_chroma_noise,
                     video_chroma_phase_noise, video_chroma_loss, composite_preemphasis):
        h, w, _ = src.shape

        fY, fI, fQ = self.rgb_to_yiq(src)
        self.composite_lowpass(fI, fQ, w, h)
        self.chroma_into_luma(fY, fI, fQ, w, h, subcarrier_amplitude)

        if composite_preemphasis != 0.0:
            self.apply_preemphasis(fY, w, h, composite_preemphasis)

        if video_noise != 0:
            self.apply_video_noise(fY, h, w, video_noise)

        self.chroma_from_luma(fY, fI, fQ, w, h, subcarrier_amplitude)

        if video_chroma_noise != 0:
            self.apply_chroma_noise(fI, fQ, h, w, video_chroma_noise)

        if video_chroma_phase_noise != 0:
            self.apply_chroma_phase_noise(fI, fQ, h, w, video_chroma_phase_noise)

        if video_chroma_loss != 0:
            # Vectorized: generate mask for all rows at once
            mask = np.random.rand(h) < video_chroma_loss
            fI[mask, :] = 0
            fQ[mask, :] = 0

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
        """Apply cascaded IIR lowpass + delay to chroma channels — vectorized per row."""
        rate = (315000000.0 * 4.0) / 88.0

        for P, cutoff, delay in [(fI, 1300000.0, 2), (fQ, 600000.0, 4)]:
            tau = 1.0 / (cutoff * 2 * np.pi)
            dt = 1.0 / rate
            alpha = dt / (tau + dt)
            filtered = _iir_lowpass_rows(P, alpha, delay=delay, passes=3)
            P[:] = filtered

    def chroma_into_luma(self, fY, fI, fQ, w, h, subcarrier_amplitude):
        """Modulate chroma into luma signal — fully vectorized."""
        pattern_u = np.tile([1, 0, -1, 0], (w + 3) // 4)[:w].astype(np.float64)
        pattern_v = np.tile([0, 1, 0, -1], (w + 3) // 4)[:w].astype(np.float64)

        chroma = fI * subcarrier_amplitude * pattern_u[np.newaxis, :]
        chroma += fQ * subcarrier_amplitude * pattern_v[np.newaxis, :]
        fY += chroma / 50.0
        fI[:] = 0
        fQ[:] = 0

    def apply_preemphasis(self, fY, w, h, composite_preemphasis):
        """Apply composite pre-emphasis highpass filter — vectorized per row."""
        rate = (315000000.0 * 4.0) / 88.0
        hz = 315000000.0 / 88.0
        tau = 1.0 / (hz * 2 * np.pi)
        dt = 1.0 / rate
        alpha = dt / (tau + dt)

        lowpassed = _iir_lowpass_rows(fY, alpha, delay=0, passes=1)
        highpassed = fY - lowpassed
        fY += highpassed * composite_preemphasis

    def apply_video_noise(self, fY, h, w, video_noise):
        """Apply correlated video noise using IIR random walk — vectorized.

        Original: noise += (randint(mod) - base); noise /= 2; fY[i] += noise
        This is: noise[n] = 0.5 * noise[n-1] + delta[n]
        """
        noise_mod = video_noise * 2 + 1
        deltas = np.random.randint(0, noise_mod, size=h * w).astype(np.float64) - video_noise
        noise_signal = _iir_noise_signal(deltas)
        fY += noise_signal.reshape(h, w)

    def apply_chroma_noise(self, fI, fQ, h, w, video_chroma_noise):
        """Apply correlated chroma noise — vectorized."""
        noise_mod = video_chroma_noise * 2 + 1

        deltas_i = np.random.randint(0, noise_mod, size=h * w).astype(np.float64) - video_chroma_noise
        deltas_q = np.random.randint(0, noise_mod, size=h * w).astype(np.float64) - video_chroma_noise

        noise_i = _iir_noise_signal(deltas_i).reshape(h, w)
        noise_q = _iir_noise_signal(deltas_q).reshape(h, w)

        fI += noise_i
        fQ += noise_q

    def apply_chroma_phase_noise(self, fI, fQ, h, w, video_chroma_phase_noise):
        """Apply chroma phase rotation per scanline — vectorized inner loop."""
        noise_mod = (video_chroma_phase_noise * 2) + 1

        # Per-row noise using IIR random walk (only h values, not h*w)
        deltas = np.random.randint(0, noise_mod, size=h).astype(np.float64) - video_chroma_phase_noise
        noise = _iir_noise_signal(deltas)

        pi_vals = (noise * np.pi) / 100.0
        sinpi = np.sin(pi_vals)[:, np.newaxis]  # (h, 1) for broadcasting
        cospi = np.cos(pi_vals)[:, np.newaxis]

        # Rotate I/Q channels per row — vectorized across width
        u = fI.copy()
        v = fQ.copy()
        fI[:] = u * cospi - v * sinpi
        fQ[:] = u * sinpi + v * cospi

    def chroma_from_luma(self, fY, fI, fQ, w, h, subcarrier_amplitude):
        """Extract chroma from composite luma signal — partially vectorized."""
        sign_pattern = np.tile([1, 1, -1, -1], (w + 3) // 4)[:w].astype(np.float64)
        even_indices = np.arange(0, w - 1, 2)
        odd_indices = np.arange(0, w - 2, 2)

        for y in range(h):
            row = fY[y].copy()

            # 4-tap moving average with lookahead
            # Pad for the delay line behavior: [0, 0, row[0], row[1], ...]
            padded = np.concatenate([np.zeros(2), row, np.zeros(2)])
            # Cumulative sum trick for box filter of width 4
            cs = np.cumsum(padded)
            # sum of 4 elements ending at position i
            smoothed = (cs[4:4 + w] - cs[:w]) / 4.0

            chroma = row - smoothed
            fY[y] = smoothed

            # Sign-flip every other pair
            chroma *= sign_pattern

            chroma = (chroma * 50) / subcarrier_amplitude

            # Assign I and Q from interleaved chroma
            fI[y, even_indices] = -chroma[even_indices]
            q_src = even_indices + 1
            valid = q_src < w
            fQ[y, even_indices[valid]] = -chroma[q_src[valid]]

            # Interpolate odd positions
            if len(odd_indices) > 0:
                next_even = np.minimum(odd_indices + 2, w - 1)
                fI[y, odd_indices + 1] = (fI[y, odd_indices] + fI[y, next_even]) / 2
                fQ[y, odd_indices + 1] = (fQ[y, odd_indices] + fQ[y, next_even]) / 2

    def render_scanlines(self, img, scale):
        h, w, _ = img.shape
        scanline_img = np.zeros_like(img)

        # Vectorized scanline pattern
        y_indices = np.arange(0, h, 3)
        scanline_img[y_indices, :, 0] = np.random.uniform(0.8, 0.9, size=(len(y_indices), 1))
        y1 = y_indices + 1
        y1 = y1[y1 < h]
        scanline_img[y1, :, 1] = np.random.uniform(0.8, 0.9, size=(len(y1), 1))
        y2 = y_indices + 2
        y2 = y2[y2 < h]
        scanline_img[y2, :, 2] = np.random.uniform(0.8, 0.9, size=(len(y2), 1))

        blurred = F.interpolate(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0),
                              scale_factor=1 / scale, mode='bilinear', align_corners=False)
        blurred = F.interpolate(blurred, size=(h, w), mode='bilinear', align_corners=False)
        blurred = blurred.squeeze().permute(1, 2, 0).numpy()

        result = img * 0.7 + blurred * 0.3 + scanline_img * 0.15
        return np.clip(result, 0, 1)
