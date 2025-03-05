# https://x.com/_pxlpshr
# https://instagram.com/pxl.pshr/

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
    
    This class uses a custom implementation of wavelet transformation to decompose an image,
    apply controlled distortion to the wavelet coefficients, and then reconstruct the image.
    
    Example usage:
        corruptor = Corruptor()
        corrupted_image, = corruptor.apply_glitch(
            image=input_tensor,
            scaling_factor_in=80.0,
            scaling_factor_out=80.0,
            noise_strength=10.0,
            color_space="HSV",
            channels_combined=True,
            wavelet_floor_mode="regular",
            wavelet_padding="edge"
        )
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "scaling_factor_in": ("FLOAT", {"default": 80.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "scaling_factor_out": ("FLOAT", {"default": 80.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "noise_strength": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "color_space": (["RGB", "HSV", "LAB", "YUV"], ),
                "channels_combined": ("BOOLEAN", {"default": True}),
                "wavelet_floor_mode": (["regular", "absolute", "threshold"], {"default": "regular"}),
                "wavelet_padding": (["edge", "constant", "reflect", "symmetric"], {"default": "edge"})
            },
            "optional": {
                "wavelet_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "noise_distribution": (["normal", "uniform", "salt_pepper"], {"default": "normal"})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_glitch"
    CATEGORY = "GlitchNodes"

    def apply_glitch(self, image, scaling_factor_in, scaling_factor_out, noise_strength, color_space, 
                    channels_combined, wavelet_floor_mode="regular", wavelet_padding="edge", 
                    wavelet_threshold=0.5, noise_distribution="normal"):
        """
        Main entry point for the corruption process.
        
        Args:
            image (torch.Tensor): Input image tensor (N,H,W,C) or (H,W,C).
                N = batch size, H = height, W = width, C = color channels
            scaling_factor_in (float): Controls corruption intensity in forward transform.
                Higher values create more subtle effects, lower values create more extreme effects.
            scaling_factor_out (float): Controls corruption intensity in reverse transform.
                Higher values amplify the effect, lower values reduce it.
            noise_strength (float): Controls the amount of random noise added.
                0.0 means no noise, higher values add more intense noise.
            color_space (str): Color space to process in ("RGB", "HSV", "LAB", "YUV").
                Different color spaces produce different visual effects.
            channels_combined (bool): Process all color channels together if True.
                When False, each channel is processed independently.
            wavelet_floor_mode (str): Method used to quantize wavelet coefficients:
                - "regular": Standard floor operation
                - "absolute": Takes absolute value before applying floor
                - "threshold": Uses threshold to clip small values
            wavelet_padding (str): Padding mode for when extending signals to power of 2:
                - "edge": Repeat edge values
                - "constant": Pad with zeros
                - "reflect": Mirror the array at boundaries
                - "symmetric": Mirror with the edge value repeated
            wavelet_threshold (float): Threshold value used when wavelet_floor_mode is "threshold"
            noise_distribution (str): Type of noise distribution to add:
                - "normal": Gaussian noise (bell curve distribution)
                - "uniform": Uniform random noise (even distribution)
                - "salt_pepper": Salt and pepper noise (random extreme values)
        
        Returns:
            tuple: Tuple containing single Tensor of corrupted image(s) in (N,H,W,C) format
        """
        try:
            # Use a single progress bar for the whole process
            if image.dim() == 4:
                # For batch processing, create one progress bar for the entire batch
                batch_size = image.shape[0]
                with tqdm(total=batch_size, desc="Processing images") as pbar:
                    results = []
                    for img in image:
                        results.append(self._process_single_image(
                            img, scaling_factor_in, scaling_factor_out, 
                            noise_strength, color_space, channels_combined,
                            wavelet_floor_mode, wavelet_padding, 
                            wavelet_threshold, noise_distribution
                        ))
                        pbar.update(1)
                result = torch.stack(results)
            else:
                result = self._process_single_image(
                    image, scaling_factor_in, scaling_factor_out,
                    noise_strength, color_space, channels_combined,
                    wavelet_floor_mode, wavelet_padding,
                    wavelet_threshold, noise_distribution
                ).unsqueeze(0)
            
            if result.shape[1] == 3:
                result = result.permute(0, 2, 3, 1)
            
            return (result,)
        except Exception as e:
            logger.error(f"Error in corruption process: {str(e)}")
            raise

    def _process_single_image(self, image, scaling_factor_in, scaling_factor_out, 
                             noise_strength, color_space, channels_combined,
                             wavelet_floor_mode, wavelet_padding, 
                             wavelet_threshold, noise_distribution):
        """
        Processes a single image through the corruption pipeline.
        Handles format conversions and applies the corruption effect.
        
        Args:
            image (torch.Tensor): Single image tensor
            Other parameters: Same as apply_glitch method
            
        Returns:
            torch.Tensor: Processed image tensor
        """
        try:
            if image.shape[0] != 3:
                image = image.permute(2, 0, 1)
            
            img_np = image.cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            
            corrupted_img = self.corrupt_image(
                img_np, scaling_factor_in, scaling_factor_out,
                noise_strength, color_space, channels_combined,
                wavelet_floor_mode, wavelet_padding,
                wavelet_threshold, noise_distribution
            )
            
            result = torch.from_numpy(corrupted_img).float() / 255.0
            result = result.permute(2, 0, 1)
            
            return result
        except Exception as e:
            logger.error(f"Error in processing single image: {str(e)}")
            raise

    def corrupt_image(self, img_np, scaling_factor_in, scaling_factor_out, 
                     noise_strength, color_space, channels_combined,
                     wavelet_floor_mode, wavelet_padding,
                     wavelet_threshold, noise_distribution):
        """
        Applies the corruption effect to the image data in the specified color space.
        
        Args:
            img_np (numpy.ndarray): Input image as NumPy array (H,W,C) in RGB format
            Other parameters: Same as apply_glitch method
            
        Returns:
            numpy.ndarray: Corrupted image as NumPy array (H,W,C) in RGB format
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
                # Process all channels as a single data stream
                raw = img_converted.reshape(-1)
                corrupted = self.process_channel(
                    raw, scaling_factor_in, scaling_factor_out,
                    wavelet_floor_mode, wavelet_padding, wavelet_threshold
                )
                corrupted = corrupted.reshape(img_converted.shape)
            else:
                # Process each channel independently
                corrupted = np.zeros_like(img_converted)
                for i in range(3):
                    channel = img_converted[:,:,i].reshape(-1)
                    corrupted[:,:,i] = self.process_channel(
                        channel, scaling_factor_in, scaling_factor_out,
                        wavelet_floor_mode, wavelet_padding, wavelet_threshold
                    ).reshape(img_converted.shape[:2])

            # Add noise with controllable strength and distribution
            if noise_strength > 0:
                if noise_distribution == "normal":
                    noise = np.random.normal(0, noise_strength, corrupted.shape)
                elif noise_distribution == "uniform":
                    noise = np.random.uniform(-noise_strength*2, noise_strength*2, corrupted.shape)
                elif noise_distribution == "salt_pepper":
                    noise = np.zeros(corrupted.shape)
                    # Salt noise
                    salt_mask = np.random.random(corrupted.shape) < (noise_strength / 200)
                    noise[salt_mask] = 255
                    # Pepper noise
                    pepper_mask = np.random.random(corrupted.shape) < (noise_strength / 200)
                    noise[pepper_mask] = -255
                
                corrupted = np.clip(corrupted + noise, 0, 255)

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

    def process_channel(self, channel, scaling_factor_in, scaling_factor_out,
                       wavelet_floor_mode, wavelet_padding, wavelet_threshold):
        """
        Processes a single channel through wavelet transformation.
        Applies forward and reverse transformations with different scaling factors.
        
        Args:
            channel (numpy.ndarray): 1D array representing image data
            scaling_factor_in (float): Scaling factor for forward transform
            scaling_factor_out (float): Scaling factor for reverse transform
            wavelet_floor_mode (str): Method used to quantize wavelet coefficients
            wavelet_padding (str): Padding mode for signal extension
            wavelet_threshold (float): Threshold value for coefficient clipping
            
        Returns:
            numpy.ndarray: Processed channel data
        """
        try:
            # Extend to power of 2 length
            n = 2 ** int(np.ceil(np.log2(len(channel))))
            padded = np.pad(channel, (0, n - len(channel)), mode=wavelet_padding)
            
            # Apply forward wavelet transform with configured parameters
            transformed = self.wtrafo(padded, scaling_factor_in, wavelet_floor_mode, wavelet_threshold)
            # Apply reverse wavelet transform
            reconstructed = self.wbtrafo(transformed, scaling_factor_out)
            
            # Return only the original length
            return reconstructed[:len(channel)]
        except Exception as e:
            logger.error(f"Error in channel processing: {str(e)}")
            raise

    def wtrafo(self, y, scaling_factor, floor_mode="regular", threshold=0.5):
        """
        Forward wavelet transformation.
        Breaks down the signal into wavelets and applies initial scaling.
        
        Args:
            y (numpy.ndarray): Input signal
            scaling_factor (float): Controls quantization level of coefficients
            floor_mode (str): Method for quantizing coefficients:
                - "regular": Standard floor operation
                - "absolute": Takes absolute value before floor
                - "threshold": Uses threshold to clip small values
            threshold (float): Threshold value for "threshold" mode
                
        Returns:
            numpy.ndarray: Wavelet coefficients
        """
        n = len(y)
        d = np.zeros(n)
        w = np.zeros(n)
        
        # First decomposition level
        a = n // 2
        w[:a] = (y[::2] - y[1::2]) * np.sqrt(0.5)
        d[:a] = (y[::2] + y[1::2]) * np.sqrt(0.5)
        
        # Additional decomposition levels
        b1, b2 = 0, a
        a //= 2
        while a > 0:
            w[b2:b2+a] = (d[b1:b1+2*a:2] - d[b1+1:b1+2*a:2]) * np.sqrt(0.5)
            d[b2:b2+a] = (d[b1:b1+2*a:2] + d[b1+1:b1+2*a:2]) * np.sqrt(0.5)
            b1, b2, a = b2, b2 + a, a // 2
        
        w[b2] = d[b1]
        
        # Apply coefficient quantization according to chosen mode
        if floor_mode == "regular":
            # Standard floor operation
            w[:-1] = np.floor(w[:-1] / scaling_factor)
            w[-1] = np.floor(w[-1] / scaling_factor + 0.5) if w[-1] > 0 else np.floor(w[-1] / scaling_factor - 0.5)
        elif floor_mode == "absolute":
            # Take absolute value before floor for more symmetric corruption
            signs = np.sign(w)
            w_abs = np.abs(w)
            w_abs = np.floor(w_abs / scaling_factor)
            w = signs * w_abs
        elif floor_mode == "threshold":
            # Apply threshold to small values for cleaner corruption
            w_scaled = w / scaling_factor
            mask = np.abs(w_scaled) < threshold
            w_scaled[mask] = 0
            w_scaled[~mask] = np.floor(w_scaled[~mask])
            w = w_scaled
        
        return w

    def wbtrafo(self, w, scaling_factor):
        """
        Reverse wavelet transformation.
        Reconstructs the signal from wavelets and applies final scaling.
        
        Args:
            w (numpy.ndarray): Wavelet coefficients
            scaling_factor (float): Amplification factor for reconstruction
                
        Returns:
            numpy.ndarray: Reconstructed signal
        """
        n = len(w)
        d = np.zeros(n)
        y = np.zeros(n)
        
        # Reconstruction through inverse wavelet transform
        d[n-2] = w[n-1]
        b1, b2 = n - 4, n - 2
        a = 1
        while a < n // 2:
            d[b1:b1+2*a:2] = (d[b2:b2+a] + w[b2:b2+a]) * np.sqrt(0.5)
            d[b1+1:b1+2*a:2] = (d[b2:b2+a] - w[b2:b2+a]) * np.sqrt(0.5)
            b2, b1, a = b1, b1 - 4*a, a * 2
        
        # Final reconstruction level
        y[::2] = (d[:a] + w[:a]) * np.sqrt(0.5)
        y[1::2] = (d[:a] - w[:a]) * np.sqrt(0.5)
        
        return y * scaling_factor