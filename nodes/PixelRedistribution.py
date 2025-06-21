# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

import math

import torch
from tqdm import tqdm


class PixelRedistribution:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "distance_mode": (["center", "top", "left", "random"], {"default": "center"}),
                "pattern": (["outward", "spiral", "waves", "diagonal"], {"default": "outward"}),
                "color_size": (
                    "INT",
                    {
                        "default": 64,
                        "min": 2,
                        "max": 256,
                        "step": 1,
                    },
                ),
                "order": ("STRING", {"default": "0,1,2"}),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.1,
                    },
                ),
                "contrast": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 4.0,
                        "step": 0.1,
                    },
                ),
                "brightness": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.1,
                    },
                ),
                "invert": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "redistribute_pixels"
    CATEGORY = "GlitchNodes"

    def adjust_contrast_brightness(self, image, contrast, brightness):
        # Apply contrast
        mean = image.mean()
        adjusted = (image - mean) * contrast + mean

        # Apply brightness
        if brightness > 0:
            adjusted = adjusted * (1 - brightness) + brightness
        else:
            adjusted = adjusted * (1 + brightness)

        return adjusted.clamp(0, 1)

    def get_adjacent_colors(self, color, searched, order, color_size):
        device = color.device
        adj_list = []

        for channel in order:
            if color[channel] > 0:
                new_color = torch.clone(color)
                new_color[channel] -= 1
                color_tuple = tuple(new_color.cpu().numpy())
                if color_tuple not in searched:
                    adj_list.append(new_color)

            if color[channel] < color_size - 1:
                new_color = torch.clone(color)
                new_color[channel] += 1
                color_tuple = tuple(new_color.cpu().numpy())
                if color_tuple not in searched:
                    adj_list.append(new_color)

        return adj_list

    def calculate_distance(self, coords, width, height, mode, pattern, strength):
        if mode == "center":
            center_x = width / 2
            center_y = height / 2
            x_diff = coords[:, 0].float() - center_x
            y_diff = coords[:, 1].float() - center_y
            base_distance = torch.sqrt(x_diff**2 + y_diff**2)

        elif mode == "top":
            base_distance = coords[:, 1].float()

        elif mode == "left":
            base_distance = coords[:, 0].float()

        elif mode == "random":
            base_distance = torch.rand_like(coords[:, 0].float())

        if pattern == "spiral":
            angle = torch.atan2(coords[:, 1].float() - height / 2, coords[:, 0].float() - width / 2)
            base_distance = base_distance + angle * width / (2 * math.pi)

        elif pattern == "waves":
            wave = torch.sin(coords[:, 0].float() * 0.1) * height * 0.25
            base_distance = base_distance + wave

        elif pattern == "diagonal":
            diagonal_component = (coords[:, 0].float() + coords[:, 1].float()) * 0.5
            base_distance = base_distance + diagonal_component

        return base_distance * strength

    def process_single_image(
        self, image, color_size, order, distance_mode, pattern, strength, contrast, brightness, invert
    ):
        device = image.device

        # Adjust contrast and brightness before processing
        process_image = self.adjust_contrast_brightness(image, contrast, brightness)

        if invert:
            process_image = 1.0 - process_image

        quantized = torch.floor(process_image * (color_size - 1))
        channels, height, width = image.shape

        transform_colorspace = {}
        points = quantized.permute(1, 2, 0)
        coords = torch.stack(
            torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"),
            dim=-1,
        )

        for h in range(height):
            for w in range(width):
                color = tuple(points[h, w].cpu().numpy())
                if color not in transform_colorspace:
                    transform_colorspace[color] = []
                transform_colorspace[color].append((w, h))

        start_keys = sorted(
            [k for k, v in transform_colorspace.items() if len(v) > 1],
            key=lambda x: len(transform_colorspace[x]),
        )

        for start_key in start_keys:
            points = torch.tensor(transform_colorspace[start_key], device=device)
            distances = self.calculate_distance(points, width, height, distance_mode, pattern, strength)
            sorted_indices = torch.argsort(distances)
            transform_colorspace[start_key] = [tuple(points[i].cpu().numpy()) for i in sorted_indices]

            queue = []
            searched = {}
            prev = {}
            end_keys = []

            start_key_tensor = torch.tensor(start_key, device=device)
            queue.append(start_key_tensor)
            searched[start_key] = None
            prev[start_key] = None

            while queue and len(end_keys) < len(transform_colorspace[start_key]) - 1:
                current_key_tensor = queue.pop(0)
                current_key = tuple(current_key_tensor.cpu().numpy())

                if current_key not in transform_colorspace or len(transform_colorspace[current_key]) == 0:
                    end_keys.append(current_key)
                    if len(end_keys) >= len(transform_colorspace[start_key]) - 1:
                        break

                adj_colors = self.get_adjacent_colors(
                    current_key_tensor,
                    searched,
                    order,
                    color_size,
                )

                for adj_color in adj_colors:
                    adj_key = tuple(adj_color.cpu().numpy())
                    searched[adj_key] = None
                    prev[adj_key] = current_key
                    queue.append(adj_color)

            for end_key in end_keys:
                if prev[end_key] is None:
                    continue

                current_key = end_key
                while prev[current_key] is not None:
                    prev_key = prev[current_key]
                    if transform_colorspace[prev_key]:
                        point = transform_colorspace[prev_key].pop(0)
                        if current_key not in transform_colorspace:
                            transform_colorspace[current_key] = []
                        transform_colorspace[current_key].append(point)
                    current_key = prev_key

        output = torch.zeros_like(image)
        for color, points in transform_colorspace.items():
            if points:
                color_tensor = torch.tensor(color, device=device).float() / (color_size - 1)
                for x, y in points:
                    output[:, y, x] = color_tensor if not invert else (1.0 - color_tensor)

        return output

    def redistribute_pixels(
        self, image, color_size, order, distance_mode, pattern, strength, contrast, brightness, invert
    ):
        try:
            if not isinstance(image, torch.Tensor):
                raise ValueError("Input image must be a torch.Tensor")

            if image.dim() != 4:
                raise ValueError(f"Expected 4D input tensor, got {image.dim()}D")

            try:
                order = [int(x.strip()) for x in order.split(",")]
                if not all(0 <= x <= 2 for x in order) or len(order) != 3:
                    raise ValueError("Order must be three comma-separated integers between 0 and 2")
            except ValueError as e:
                raise ValueError(f"Invalid order format: {e}")

            if image.max() > 1.0:
                image = image / 255.0

            batch_size = image.shape[0]
            processed_images = []

            pbar = tqdm(total=batch_size, desc="Processing images")
            for i in range(batch_size):
                processed_image = self.process_single_image(
                    image[i],
                    color_size,
                    order,
                    distance_mode,
                    pattern,
                    strength,
                    contrast,
                    brightness,
                    invert,
                )
                processed_images.append(processed_image)
                pbar.update(1)
            pbar.close()

            output = torch.stack(processed_images, dim=0)
            output = (output * 255.0).clamp(0, 255)

            return (output,)

        except Exception as e:
            print(f"Error in PixelRedistribution: {e!s}")
            raise e
