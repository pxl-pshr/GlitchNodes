import torch
import numpy as np
import logging
from tqdm import tqdm
import cv2

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Corruptor:
    """
    A node that applies controlled corruption effects to images using wavelet transformations.
    The corruption can be applied in multiple color spaces with adjustable intensity.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "scaling_factor_in": ("FLOAT", {"default": 80.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "scaling_factor_out": ("FLOAT", {"default": 80.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "noise_strength": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "color_space": (["RGB", "HSV", "LAB", "YUV"],),
                "channels_combined": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_glitch"
    CATEGORY = "image/processing"

    def apply_glitch(self, image, scaling_factor_in, scaling_factor_out, noise_strength, color_space, channels_combined):
        """
        Main entry point for the corruption process.
        
        Args:
            image: Input image tensor (N,H,W,C) or (H,W,C)
            scaling_factor_in: Controls corruption intensity in forward transform
            scaling_factor_out: Controls corruption intensity in reverse transform
            noise_strength: Controls the amount of random noise added
            color_space: Color space to process in ("RGB", "HSV", "LAB", "YUV")
            channels_combined: Process all color channels together if True
        
        Returns:
            Corrupted image tensor in (N,H,W,C) format
        """
        try:
            if image.dim() == 4:
                results = []
                for img in tqdm(image, desc="Processing batch"):
                    results.append(self._process_single_image(
                        img, scaling_factor_in, scaling_factor_out, 
                        noise_strength, color_space, channels_combined
                    ))
                result = torch.stack(results)
            else:
                result = self._process_single_image(
                    image, scaling_factor_in, scaling_factor_out,
                    noise_strength, color_space, channels_combined
                ).unsqueeze(0)
            
            if result.shape[1] == 3:
                result = result.permute(0, 2, 3, 1)
            
            return (result,)
        except Exception as e:
            logger.error(f"Error in corruption process: {str(e)}")
            raise

    def _process_single_image(self, image, scaling_factor_in, scaling_factor_out, noise_strength, color_space, channels_combined):
        """
        Processes a single image through the corruption pipeline.
        Handles format conversions and applies the corruption effect.
        """
        try:
            if image.shape[0] != 3:
                image = image.permute(2, 0, 1)
            
            img_np = image.cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            
            corrupted_img = self.corrupt_image(
                img_np, scaling_factor_in, scaling_factor_out,
                noise_strength, color_space, channels_combined
            )
            
            result = torch.from_numpy(corrupted_img).float() / 255.0
            result = result.permute(2, 0, 1)
            
            return result
        except Exception as e:
            logger.error(f"Error in processing single image: {str(e)}")
            raise

    def corrupt_image(self, img_np, scaling_factor_in, scaling_factor_out, noise_strength, color_space, channels_combined):
        """
        Applies the corruption effect to the image data in the specified color space.
        """
        try:
            # Convert to target color space
            if color_space == "HSV":
                img_converted = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            elif color_space == "LAB":
                img_converted = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            elif color_space == "YUV":
                img_converted = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
            else:  # RGB
                img_converted = img_np.copy()

            if channels_combined:
                raw = img_converted.reshape(-1)
                corrupted = self.process_channel(raw, scaling_factor_in, scaling_factor_out)
                corrupted = corrupted.reshape(img_converted.shape)
            else:
                corrupted = np.zeros_like(img_converted)
                for i in tqdm(range(3), desc="Processing channels"):
                    channel = img_converted[:,:,i].reshape(-1)
                    corrupted[:,:,i] = self.process_channel(
                        channel, scaling_factor_in, scaling_factor_out
                    ).reshape(img_converted.shape[:2])

            # Add noise with controllable strength
            if noise_strength > 0:
                corrupted = np.clip(
                    corrupted + np.random.normal(0, noise_strength, corrupted.shape),
                    0, 255
                )

            # Convert back to RGB
            if color_space == "HSV":
                corrupted = cv2.cvtColor(corrupted.astype(np.uint8), cv2.COLOR_HSV2RGB)
            elif color_space == "LAB":
                corrupted = cv2.cvtColor(corrupted.astype(np.uint8), cv2.COLOR_LAB2RGB)
            elif color_space == "YUV":
                corrupted = cv2.cvtColor(corrupted.astype(np.uint8), cv2.COLOR_YUV2RGB)

            return np.clip(corrupted, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error in image corruption: {str(e)}")
            raise

    def process_channel(self, channel, scaling_factor_in, scaling_factor_out):
        """
        Processes a single channel through wavelet transformation.
        Applies forward and reverse transformations with different scaling factors.
        """
        try:
            n = 2 ** int(np.ceil(np.log2(len(channel))))
            padded = np.pad(channel, (0, n - len(channel)), mode='edge')
            
            transformed = self.wtrafo(padded, scaling_factor_in)
            reconstructed = self.wbtrafo(transformed, scaling_factor_out)
            
            return reconstructed[:len(channel)]
        except Exception as e:
            logger.error(f"Error in channel processing: {str(e)}")
            raise

    def wtrafo(self, y, scaling_factor):
        """
        Forward wavelet transformation.
        Breaks down the signal into wavelets and applies initial scaling.
        """
        n = len(y)
        d = np.zeros(n)
        w = np.zeros(n)
        
        a = n // 2
        w[:a] = (y[::2] - y[1::2]) * np.sqrt(0.5)
        d[:a] = (y[::2] + y[1::2]) * np.sqrt(0.5)
        
        b1, b2 = 0, a
        a //= 2
        while a > 0:
            w[b2:b2+a] = (d[b1:b1+2*a:2] - d[b1+1:b1+2*a:2]) * np.sqrt(0.5)
            d[b2:b2+a] = (d[b1:b1+2*a:2] + d[b1+1:b1+2*a:2]) * np.sqrt(0.5)
            b1, b2, a = b2, b2 + a, a // 2
        
        w[b2] = d[b1]
        
        w[:-1] = np.floor(w[:-1] / scaling_factor)
        w[-1] = np.floor(w[-1] / scaling_factor + 0.5) if w[-1] > 0 else np.floor(w[-1] / scaling_factor - 0.5)
        
        return w

    def wbtrafo(self, w, scaling_factor):
        """
        Reverse wavelet transformation.
        Reconstructs the signal from wavelets and applies final scaling.
        """
        n = len(w)
        d = np.zeros(n)
        y = np.zeros(n)
        
        d[n-2] = w[n-1]
        b1, b2 = n - 4, n - 2
        a = 1
        while a < n // 2:
            d[b1:b1+2*a:2] = (d[b2:b2+a] + w[b2:b2+a]) * np.sqrt(0.5)
            d[b1+1:b1+2*a:2] = (d[b2:b2+a] - w[b2:b2+a]) * np.sqrt(0.5)
            b2, b1, a = b1, b1 - 4*a, a * 2
        
        y[::2] = (d[:a] + w[:a]) * np.sqrt(0.5)
        y[1::2] = (d[:a] - w[:a]) * np.sqrt(0.5)
        
        return y * scaling_factor