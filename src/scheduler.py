import numpy as np
import torch
from noise import NoiseSampler

# https://huggingface.co/blog/annotated-diffusion


# This class should be inhereting from NoiseSampler
class CosineScheduler(NoiseSampler):
    """Scheduler for discrete temporal values in Diffusion Model training.
    """

    def __init__(self, timesteps, s=0.008):
        super().__init__(timesteps, s)
    
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

    
    
