import torch
import numpy as np
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Corruptor:
    """
    A node that applies controlled corruption effects to images using wavelet transformations.
    The corruption can be applied in both RGB and HSB color spaces, with adjustable intensity
    through scaling factors for both the forward and backward transformations.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "scaling_factor_in": ("FLOAT", {"default": 80.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "scaling_factor_out": ("FLOAT", {"default": 80.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "do_hsb": ("BOOLEAN", {"default": False}),
                "channels_combined": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_glitch"
    CATEGORY = "image/processing"

    def apply_glitch(self, image, scaling_factor_in, scaling_factor_out, do_hsb, channels_combined):
        """
        Main entry point for the corruption process.
        
        Args:
            image: Input image tensor (N,H,W,C) or (H,W,C)
            scaling_factor_in: Controls corruption intensity in forward transform
            scaling_factor_out: Controls corruption intensity in reverse transform
            do_hsb: Process in HSB color space if True, RGB if False
            channels_combined: Process all color channels together if True
        
        Returns:
            Corrupted image tensor in (N,H,W,C) format
        """
        try:
            if image.dim() == 4:
                results = []
                for img in tqdm(image, desc="Processing batch"):
                    results.append(self._process_single_image(img, scaling_factor_in, scaling_factor_out, do_hsb, channels_combined))
                result = torch.stack(results)
            else:
                result = self._process_single_image(image, scaling_factor_in, scaling_factor_out, do_hsb, channels_combined).unsqueeze(0)
            
            if result.shape[1] == 3:
                result = result.permute(0, 2, 3, 1)
            
            return (result,)
        except Exception as e:
            logger.error(f"Error in corruption process: {str(e)}")
            raise

    def _process_single_image(self, image, scaling_factor_in, scaling_factor_out, do_hsb, channels_combined):
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
            
            corrupted_img = self.corrupt_image(img_np, scaling_factor_in, scaling_factor_out, do_hsb, channels_combined)
            
            result = torch.from_numpy(corrupted_img).float() / 255.0
            result = result.permute(2, 0, 1)
            
            return result
        except Exception as e:
            logger.error(f"Error in processing single image: {str(e)}")
            raise

    def corrupt_image(self, img_np, scaling_factor_in, scaling_factor_out, do_hsb, channels_combined):
        """
        Applies the corruption effect to the image data.
        Can process in RGB or HSB color space, with channels either combined or separate.
        """
        try:
            if do_hsb:
                img_np = self.rgb_to_hsb(img_np)

            if channels_combined:
                raw = img_np.reshape(-1)
                corrupted = self.process_channel(raw, scaling_factor_in, scaling_factor_out)
                corrupted = corrupted.reshape(img_np.shape)
            else:
                corrupted = np.zeros_like(img_np)
                for i in tqdm(range(3), desc="Processing channels"):
                    channel = img_np[:,:,i].reshape(-1)
                    corrupted[:,:,i] = self.process_channel(channel, scaling_factor_in, scaling_factor_out).reshape(img_np.shape[:2])

            corrupted = np.clip(corrupted + np.random.normal(0, 10, corrupted.shape), 0, 255)

            if do_hsb:
                corrupted = self.hsb_to_rgb(corrupted)

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

    def rgb_to_hsb(self, rgb):
        """
        Converts RGB color space to HSB (HSV).
        Used when do_hsb is True to process in HSB color space.
        """
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        max_val = np.max(rgb, axis=2)
        min_val = np.min(rgb, axis=2)
        diff = max_val - min_val
        
        h = np.zeros_like(r)
        s = np.zeros_like(r)
        v = max_val
        
        h[max_val == r] = (60 * ((g - b) / diff) % 360)[max_val == r]
        h[max_val == g] = (120 + 60 * ((b - r) / diff))[max_val == g]
        h[max_val == b] = (240 + 60 * ((r - g) / diff))[max_val == b]
        h[diff == 0] = 0
        
        s[max_val != 0] = (diff / max_val)[max_val != 0]
        
        return np.stack([h, s, v], axis=2)

    def hsb_to_rgb(self, hsb):
        """
        Converts HSB (HSV) color space back to RGB.
        Used when do_hsb is True to convert back after processing.
        """
        h, s, v = hsb[:,:,0], hsb[:,:,1], hsb[:,:,2]
        h = h / 360.0
        
        i = np.floor(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        i = i.astype(int) % 6
        
        rgb = np.zeros_like(hsb)
        rgb[i == 0] = np.dstack((v, t, p))[i == 0]
        rgb[i == 1] = np.dstack((q, v, p))[i == 1]
        rgb[i == 2] = np.dstack((p, v, t))[i == 2]
        rgb[i == 3] = np.dstack((p, q, v))[i == 3]
        rgb[i == 4] = np.dstack((t, p, v))[i == 4]
        rgb[i == 5] = np.dstack((v, p, q))[i == 5]
        
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)