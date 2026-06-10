# Original Styles from https://github.com/Datamosh-js/datamosh
# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

class Rekked:
    """Applies various datamosh-inspired glitch effects with multiple artistic modes."""
    MODES = ["blurbobb", "fatcat", "vaporwave", "castles", "chimera", "gazette",
             "manticore95", "schifty", "vana", "veneneux", "void", "walter"]

    def __init__(self):
        self.modes = {name: getattr(self, name) for name in self.MODES}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_rekked"
    CATEGORY = "GlitchNodes"
    DESCRIPTION = "Applies datamosh-inspired glitch effects with modes like vaporwave, chimera, void, and more"

    def apply_rekked(self, image, mode, seed):
        logger.info(f"Applying Rekked effect: {mode}")
        try:
            np_image = image.cpu().numpy().astype(np.float32)

            if np_image.shape[-1] == 1:
                np_image = np.repeat(np_image, 3, axis=-1)

            batch, height, width, channels = np_image.shape

            processed = []
            for i in range(batch):
                rng = np.random.default_rng(seed + i)
                flat_image = np_image[i].reshape(-1, channels).copy()
                moshed = self.modes[mode](flat_image, width, height, rng)
                processed.append(moshed.reshape(height, width, channels))

            moshed_image = np.clip(np.stack(processed, axis=0), 0, 1).astype(np.float32)
            moshed_image = torch.from_numpy(moshed_image).to(image.device)

            logger.info("Rekked effect completed")
            return (moshed_image,)
        except Exception as e:
            logger.error(f"Error in Rekked processing: {str(e)}")
            raise

    def blurbobb(self, data, width, height, rng):
        # Vectorized run-length version of the original per-pixel counter loop:
        # randomize while counter < 64, pass through until counter exceeds 128,
        # then reset the counter to a random value.
        n = data.shape[0]
        mask = np.zeros(n, dtype=bool)
        i = 0
        counter = 0
        while i < n:
            if counter < 64:
                run = min(64 - counter, n - i)
                mask[i:i + run] = True
                i += run
                counter += run
            else:
                run = min(129 - counter, n - i)
                i += run
                counter += run
                if counter > 128:
                    counter = int(rng.integers(128))
        data[mask, :3] = rng.random((int(mask.sum()), 3), dtype=np.float32)
        return data

    def fatcat(self, data, width, height, rng):
        for _ in range(4):
            data[:, :3] = np.minimum(data[:, :3] * 1.4, 1.0)
        return data

    def vaporwave(self, data, width, height, rng):
        COLORS = np.array([
            [0, 184/255, 1],
            [1, 0, 193/255],
            [150/255, 0, 1],
            [0, 1, 249/255]
        ])

        rgb = data[:, :3]
        conditions = [
            (rgb <= 15/255),
            (rgb > 15/255) & (rgb <= 60/255),
            (rgb > 60/255) & (rgb <= 120/255),
            (rgb > 120/255) & (rgb <= 180/255),
            (rgb > 180/255) & (rgb <= 234/255),
            (rgb >= 235/255)
        ]

        choices = [
            [0, 0, 0],
            COLORS[0],
            COLORS[1],
            COLORS[2],
            COLORS[3],
            [1, 1, 1]
        ]

        data[:, :3] = np.select(conditions, choices, rgb).astype(np.float32)
        return data

    def castles(self, data, width, height, rng):
        high, low = 165/255, 80/255
        rgb = data[:, :3]
        mask = (rgb < high) & (rgb > low)
        rgb[~mask] = 0
        return data

    def chimera(self, data, width, height, rng):
        noise_threshold = 0.2
        grain_threshold = 0.4

        mix = np.array([[1, 0.5, 0.25], [0.5, 1, 0.25], [0.25, 0.5, 1]], dtype=np.float32)
        data[:, :3] = data[:, :3] @ mix

        rgb = data[:, :3]

        # Add noise, darken, and add grain
        noise = rng.random(rgb.shape) < noise_threshold
        grain = rng.random(rgb.shape) < grain_threshold

        rgb[noise] += rng.integers(1, 16, size=int(noise.sum())) / 255
        rgb -= rng.integers(0, 31, size=rgb.shape) / 255
        rgb[grain] += rng.integers(0, 51, size=int(grain.sum())) / 255

        data[:, :3] = np.clip(rgb, 0, 1)
        return data

    def gazette(self, data, width, height, rng):
        # Vectorized over 4-pixel groups; every third group keeps the original
        # pixels, the rest collapse to a single luma-derived value covering all
        # 4 pixels (no stranded black pixels). Alpha passes through unchanged.
        ret = data.copy()
        n_groups = data.shape[0] // 4
        if n_groups == 0:
            return ret

        first = data[:n_groups * 4:4, :3]
        max_val = first.max(axis=1)
        min_val = first.min(axis=1)
        L = first.mean(axis=1)

        value = np.where(
            L > 0.65, 1.0,
            np.where(
                L < 0.35, 0.0,
                np.where(rng.random(n_groups) > 0.5, max_val, min_val)
            )
        ).astype(np.float32)

        keep = (np.arange(n_groups) % 3) == 0
        groups = ret[:n_groups * 4].reshape(n_groups, 4, -1)
        groups[~keep, :, :3] = value[~keep, np.newaxis, np.newaxis]

        return ret

    def manticore95(self, data, width, height, rng):
        def limiter(x, min_val):
            return max(x, min_val)

        def get_closest_root(x):
            # Align offsets to 4-pixel groups so the scatter/skip artifacts stay
            # column-aligned (keeps the blocky aesthetic consistent).
            return x - (x % 4)

        def max_offset(x):
            return np.argmax(x), np.max(x)

        original_shape = data.shape
        n = data.shape[0]
        sq_len = int(np.sqrt(n) / 8)
        ret = np.zeros_like(data)
        i = 0
        out_i = 0

        has_alpha = data.shape[1] == 4

        while i < n and out_i < n:
            size = int(limiter(rng.random() * (width / 40), 1))
            offset, max_val = max_offset(data[i, :3])
            skip = get_closest_root(int(rng.random() * sq_len))

            copy = min(size, n - i, n - out_i)
            ret[out_i:out_i + copy, offset] = data[i:i + copy, offset]
            out_i += copy
            i += copy

            out_i += skip
            i += skip

        y_axises_count = int(np.sqrt(n) * 4)
        ks = np.arange(20)
        for _ in range(y_axises_count):
            swap_from = get_closest_root(int(rng.random() * n))
            if swap_from < n - width * 64:
                for j in range(3):
                    swap_paths = swap_from + j + width * 4 * (ks - 4)
                    valid = (swap_paths >= 0) & (swap_paths < n)
                    ret[swap_paths[valid], j] = ret[swap_from, j]

        if has_alpha:
            ret[:, 3] = data[:, 3]

        # Ensure the output has the same shape as the input
        ret = ret[:original_shape[0], :original_shape[1]]

        return ret

    def schifty(self, data, width, height, rng):
        # Datamosh-style chunk displacement: copy each chunk to a randomly
        # shifted destination so rows smear and tear.
        n = data.shape[0]
        result = data.copy()
        index = 0

        while index < n:
            size = max(int(rng.random() * 1024 * 4), 1)
            size = min(size, n - index)

            shift = int(rng.integers(1, max(width * 8, 2)))
            dest = (index + shift) % n
            count = min(size, n - dest)
            result[dest:dest + count] = data[index:index + count]

            index += size

        return result

    def vana(self, data, width, height, rng):
        def give_seed():
            seed = np.zeros(3)
            ind1, ind2 = rng.choice(3, 2, replace=False)
            seed[ind1] = max(rng.random(), 0.3)
            if rng.random() > 0.5:
                seed[ind2] = max(rng.random(), 0.3)
            return seed

        seed = give_seed()

        # Apply the effect with more controlled scaling
        data[:, 0] = np.clip(data[:, 0] * seed[0] + 0.1 * seed[2], 0, 1)  # Red
        data[:, 1] = np.clip(data[:, 1] * seed[1] + 0.1 * seed[0], 0, 1)  # Green
        data[:, 2] = np.clip(data[:, 2] * seed[2] + 0.1 * seed[1], 0, 1)  # Blue

        # Normalize to prevent any channel from dominating
        max_vals = np.max(data[:, :3], axis=1, keepdims=True)
        data[:, :3] = data[:, :3] / (max_vals + 1e-8)

        # Add some randomness to break up solid colors
        noise = rng.random(data[:, :3].shape) * 0.1
        data[:, :3] = np.clip(data[:, :3] + noise, 0, 1)

        return data

    def veneneux(self, data, width, height, rng):
        def give_seed():
            seed = np.zeros(3)
            ind1, ind2 = rng.choice(3, 2, replace=False)
            seed[ind1] = max(rng.random(), 0.1)
            if rng.random() > 0.5:
                seed[ind2] = max(rng.random(), 0.1)
            return seed

        seed = give_seed()
        seed_change = 2
        for i in range(0, data.shape[0], width):
            seed_change -= 1
            if seed_change == 0:
                seed = give_seed()
                seed_change = int(rng.random() * height / 4)

            data[i:i+width, 0] = (data[i:i+width, 0] * seed[0] + seed[2]) % 1.0
            data[i:i+width, 1] = (data[i:i+width, 1] * seed[1] + seed[0]) % 1.0
            data[i:i+width, 2] = (data[i:i+width, 2] * seed[2] + seed[0]) % 1.0

        return data

    def void(self, data, width, height, rng):
        noise_threshold = 0.2
        grain_threshold = 0.4

        rgb = data[:, :3]

        noise = rng.random(rgb.shape) < noise_threshold
        grain = rng.random(rgb.shape) < grain_threshold

        rgb -= rng.integers(1, 16, rgb.shape) / 255
        rgb[rgb < 0] += 1

        rgb[noise] += rng.integers(1, 16, int(noise.sum())) / 255
        rgb -= rng.integers(0, 41, rgb.shape) / 255
        rgb[grain] += rng.integers(0, 51, int(grain.sum())) / 255

        data[:, :3] = np.clip(rgb, 0, 1)
        return data

    def walter(self, data, width, height, rng):
        # Generate color thresholds with better distribution
        def balanced_seed():
            # Generate values between 0.2 and 0.8 to avoid extreme values
            return rng.uniform(0.2, 0.8)

        # Create threshold arrays with balanced values
        hurp = np.array([balanced_seed() for _ in range(3)])
        lurp = np.array([balanced_seed() for _ in range(3)])

        # Ensure lurp is always lower than hurp
        lurp, hurp = np.minimum(lurp, hurp), np.maximum(lurp, hurp)

        # Calculate a balanced multiplier for each channel
        multipliers = rng.uniform(0.3, 0.7, size=3)

        # Process each channel with individual characteristics
        for i in range(3):
            mask_low = data[:, i] < lurp[i]
            mask_high = data[:, i] > hurp[i]
            mask = mask_low | mask_high

            # Apply transformation with channel-specific multiplier
            data[mask, i] = np.clip(
                (hurp[i] - lurp[i]) * multipliers[i] + data[mask, i] * multipliers[i],
                0, 1
            )

        # Apply color balance correction
        # Calculate the mean intensity for each channel
        channel_means = np.mean(data[:, :3], axis=0)

        # Calculate correction factors to balance the channels
        max_mean = np.max(channel_means)
        if max_mean > 0:
            correction_factors = 0.5 * (1 + channel_means / max_mean)

            # Apply correction while maintaining the artistic effect
            for i in range(3):
                data[:, i] = np.clip(data[:, i] * correction_factors[i], 0, 1)

        # Add subtle noise to break up solid colors
        noise = rng.uniform(-0.05, 0.05, size=data[:, :3].shape)
        data[:, :3] = np.clip(data[:, :3] + noise, 0, 1)

        return data
