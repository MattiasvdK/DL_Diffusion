import numpy as np
import torch
from noise import NoiseSampler

# https://huggingface.co/blog/annotated-diffusion


# This class should be inhereting from NoiseSampler
class CosineScheduler():
    """Scheduler for discrete temporal values in Diffusion Model training.
    """

    def __init__(self, timesteps, s=0.008):
        self.timesteps = timesteps
        time = torch.linspace(0, timesteps, timesteps + 1)
        alphas = alphas = torch.cos((time / timesteps + s) / (1 + s) * np.pi / 2)**2
        alphas = alphas / alphas[0]
        alphas = alphas[1:] / alphas[:-1]
        self.betas = torch.clip(1 - alphas, 0.0001, 0.9999)

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_inv_sqrt = torch.sqrt(1 / self.alphas)
        self.alphas_cumprod_sqrt = torch.sqrt(self.alphas_cumprod)
        self.alphas_cumprod_min_sqrt = torch.sqrt(1 - self.alphas_cumprod)
        
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        self.posterior_variance = self.betas * (1 - alphas_cumprod_prev) / (1 - self.alphas_cumprod)
    
    def __call__(self, img, timestep):
        """Noise image based on schedule."""
        return self.apply_noise(img, timestep)
    
    def apply_noise(self, img, timestep):
        """Apply noise to image.
        
        Returns:
            Image tensor with noise
            Noise tensor for loss calculation"""
        noise = self._gaussian_noise(img.shape)
        sqrt_alpha = self._obtain(self.alphas_cumprod_sqrt, timestep, img.shape)
        sqrt_one_minus_alpha = self._obtain(self.alphas_cumprod_min_sqrt, timestep, img.shape)
        return sqrt_alpha * img + sqrt_one_minus_alpha * noise, noise

    
    def _obtain(self, source, timestep, target_shape):
        """Obtain values from target in timestep index for batches.
        Based on the extract function from: https://huggingface.co/blog/annotated-diffusion.
        """
        batch_size = timestep.shape[0]
        values = source.gather(-1, timestep.cpu())
        return values.reshape(batch_size, *((1,) * (len(target_shape) - 1))).to(timestep.device)
    
    def _gaussian_noise(self, shape):
        """Gaussian noise for sampling."""
        return torch.randn(shape)
    
    