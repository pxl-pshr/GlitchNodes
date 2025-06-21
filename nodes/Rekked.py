# Original Styles from https://github.com/Datamosh-js/datamosh
# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import numpy as np
import torch


class Rekked:
    def __init__(self):
        self.modes = {
            "blurbobb": self.blurbobb,
            "fatcat": self.fatcat,
            "vaporwave": self.vaporwave,
            "castles": self.castles,
            "chimera": self.chimera,
            "gazette": self.gazette,
            "manticore95": self.manticore95,
            "schifty": self.schifty,
            "vana": self.vana,
            "veneneux": self.veneneux,
            "void": self.void,
            "walter": self.walter,
        }

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "mode": (list(cls().modes.keys()),)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_Rekked"
    CATEGORY = "GlitchNodes"

    def apply_Rekked(self, image, mode):
        # Convert to numpy array
        np_image = image.cpu().numpy()

        # Get image dimensions
        batch, height, width, channels = np_image.shape

        # Reshape to 2D array
        flat_image = np_image.reshape(-1, channels)

        # Apply the selected mode
        moshed_image = self.modes[mode](flat_image.copy(), width, height)

        # Reshape back to original dimensions
        moshed_image = moshed_image.reshape(batch, height, width, channels)

        # Convert back to torch tensor
        moshed_image = torch.from_numpy(moshed_image).to(image.device)

        return (moshed_image,)

    def blurbobb(self, data, width, height):
        counter = 0
        for i in range(data.shape[0]):
            if counter < 64:
                data[i] = np.random.rand(data.shape[1])

            counter += 1
            if counter > 128:
                counter = np.random.randint(128)
        return data

    def fatcat(self, data, width, height):
        for _ in range(4):
            data = np.minimum(data * 1.4, 1.0)
        return data

    def vaporwave(self, data, width, height):
        COLORS = np.array([[0, 184 / 255, 1], [1, 0, 193 / 255], [150 / 255, 0, 1], [0, 1, 249 / 255]])

        conditions = [
            (data <= 15 / 255),
            (data > 15 / 255) & (data <= 60 / 255),
            (data > 60 / 255) & (data <= 120 / 255),
            (data > 120 / 255) & (data <= 180 / 255),
            (data > 180 / 255) & (data <= 234 / 255),
            (data >= 235 / 255),
        ]

        choices = [[0, 0, 0], COLORS[0], COLORS[1], COLORS[2], COLORS[3], [1, 1, 1]]

        return np.select(conditions, choices, data)

    def castles(self, data, width, height):
        high, low = 165 / 255, 80 / 255
        mask = (data < high) & (data > low)
        data[~mask] = 0
        return data

    def chimera(self, data, width, height):
        noise_threshold = 0.2
        grain_threshold = 0.4
        chimera_weight = [0.25, 0.5]

        # Chimera effect
        for y in range(height):
            for x in range(width):
                index = y * width + x
                r, g, b = data[index, :3]
                data[index, 0] = r + g * chimera_weight[1] + b * chimera_weight[0]
                data[index, 1] = r * chimera_weight[1] + g + b * chimera_weight[0]
                data[index, 2] = r * chimera_weight[0] + g * chimera_weight[1] + b

        # Add noise, darken, and add grain
        noise = np.random.random(data.shape) < noise_threshold
        grain = np.random.random(data.shape) < grain_threshold

        data[noise] += np.random.randint(1, 16, size=data[noise].shape) / 255
        data -= np.random.randint(0, 31, size=data.shape) / 255
        data[grain] += np.random.randint(0, 51, size=data[grain].shape) / 255

        return np.clip(data, 0, 1)

    def gazette(self, data, width, height):
        has_alpha = data.shape[1] == 4
        ret = np.zeros_like(data)

        for i in range(0, data.shape[0], 4):
            if i % 12 == 0:
                ret[i : i + 4] = data[i : i + 4]
            else:
                r, g, b = data[i, :3]
                max_val = np.max([r, g, b])
                min_val = np.min([r, g, b])
                L = np.mean([r, g, b])

                if L > 0.65:
                    value = 1
                elif L < 0.35:
                    value = 0
                else:
                    value = max_val if np.random.random() > 0.5 else min_val

                ret[i : i + 3, :3] = value
                if has_alpha:
                    ret[i : i + 3, 3] = 1  # Alpha channel

        return ret

    def manticore95(self, data, width, height):
        def limiter(x, min_val):
            return max(x, min_val)

        def get_closest_root(x):
            return x - (x % 4)

        def max_offset(x):
            return np.argmax(x), np.max(x)

        original_shape = data.shape
        sq_len = int(np.sqrt(data.shape[0]) / 8)
        ret = np.zeros_like(data)
        i = 0
        out_i = 0

        has_alpha = data.shape[1] == 4

        while i < data.shape[0] and out_i < data.shape[0]:
            size = int(limiter(np.random.random() * (width / 40), 1))
            offset, max_val = max_offset(data[i, :3])
            skip = get_closest_root(int(np.random.random() * sq_len))

            for _ in range(size):
                if out_i < data.shape[0]:
                    ret[out_i, offset] = data[i, offset]
                    if has_alpha:
                        ret[out_i, 3] = 1  # Alpha channel
                    out_i += 1
                i += 1

            out_i += skip
            i += skip

        y_axises_count = int(np.sqrt(data.shape[0]) * 4)
        for _ in range(y_axises_count):
            swap_from = get_closest_root(int(np.random.random() * data.shape[0]))
            if swap_from < data.shape[0] - width * 64:
                for j in range(3):
                    for k in range(20):
                        swap_path = swap_from + j + width * 4 * (k - 4)
                        if 0 <= swap_path < data.shape[0]:
                            ret[swap_path, j] = ret[swap_from, j]

        # Ensure the output has the same shape as the input
        ret = ret[: original_shape[0], : original_shape[1]]

        return ret

    def schifty(self, data, width, height):
        original_size = data.shape[0]
        channels = data.shape[1]
        result = np.zeros_like(data)
        index = 0

        while index < original_size:
            size = int(np.random.random() * 1024 * 4)
            size = min(size, original_size - index)

            chunk = data[index : index + size]
            result[index : index + size] = chunk

            index += size

        return result[:original_size]

    def vana(self, data, width, height):
        def give_seed():
            seed = np.zeros(3)
            ind1, ind2 = np.random.choice(3, 2, replace=False)
            seed[ind1] = max(np.random.random(), 0.3)
            if np.random.random() > 0.5:
                seed[ind2] = max(np.random.random(), 0.3)
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
        noise = np.random.random(data[:, :3].shape) * 0.1
        data[:, :3] = np.clip(data[:, :3] + noise, 0, 1)

        return data

    def veneneux(self, data, width, height):
        def give_seed():
            seed = np.zeros(3)
            ind1, ind2 = np.random.choice(3, 2, replace=False)
            seed[ind1] = max(np.random.random(), 0.1)
            if np.random.random() > 0.5:
                seed[ind2] = max(np.random.random(), 0.1)
            return seed

        seed = give_seed()
        seed_change = 2
        for i in range(0, data.shape[0], width):
            seed_change -= 1
            if seed_change == 0:
                seed = give_seed()
                seed_change = int(np.random.random() * height / 4)

            data[i : i + width, 0] = (data[i : i + width, 0] * seed[0] + seed[2]) % 1.0
            data[i : i + width, 1] = (data[i : i + width, 1] * seed[1] + seed[1]) % 1.0
            data[i : i + width, 2] = (data[i : i + width, 2] * seed[2] + seed[0]) % 1.0
            if data.shape[1] == 4:
                data[i : i + width, 3] = np.random.random(width)

        return data

    def void(self, data, width, height):
        noise_threshold = 0.2
        grain_threshold = 0.4

        noise = np.random.random(data.shape) < noise_threshold
        grain = np.random.random(data.shape) < grain_threshold

        data -= np.random.randint(1, 16, data.shape) / 255
        data[data < 0] += 1

        data[noise] += np.random.randint(1, 16, data[noise].shape) / 255
        data -= np.random.randint(0, 41, data.shape) / 255
        data[grain] += np.random.randint(0, 51, data[grain].shape) / 255

        return np.clip(data, 0, 1)

    def walter(self, data, width, height):
        # Generate color thresholds with better distribution
        def balanced_seed():
            # Generate values between 0.2 and 0.8 to avoid extreme values
            return np.random.uniform(0.2, 0.8)

        # Create threshold arrays with balanced values
        hurp = np.array([balanced_seed() for _ in range(3)])
        lurp = np.array([balanced_seed() for _ in range(3)])

        # Ensure lurp is always lower than hurp
        lurp, hurp = np.minimum(lurp, hurp), np.maximum(lurp, hurp)

        # Calculate a balanced multiplier for each channel
        multipliers = np.random.uniform(0.3, 0.7, size=3)

        # Process each channel with individual characteristics
        for i in range(3):
            mask_low = data[:, i] < lurp[i]
            mask_high = data[:, i] > hurp[i]
            mask = mask_low | mask_high

            # Apply transformation with channel-specific multiplier
            data[mask, i] = np.clip((hurp[i] - lurp[i]) * multipliers[i] + data[mask, i] * multipliers[i], 0, 1)

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
        noise = np.random.uniform(-0.05, 0.05, size=data[:, :3].shape)
        data[:, :3] = np.clip(data[:, :3] + noise, 0, 1)

        # Preserve alpha channel if it exists
        if data.shape[1] == 4:
            data[:, 3] = 1.0

        return data
