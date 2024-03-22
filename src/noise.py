import numpy as np
import torch

# https://huggingface.co/blog/annotated-diffusion

# Cosine schedule function

def cosine_schedule(timesteps, s=0.008):
    # Cosine schedule from: `Improved Denoising Diffusion Probabilistic Models`
    # Based on code from: https://huggingface.co/blog/annotated-diffusion
    time = torch.linspace(0, timesteps, timesteps+1)
    alphas = torch.cos((time / timesteps + s) / (1 + s) * (np.pi / 2))**2
    alphas = alphas / alphas[0]
    alphas = alphas[1:] / alphas[:-1]
    betas = 1 - alphas
    return torch.clip(betas, 0, 0.999)


class NoiseSampler():
    def __init__(self, timesteps, noise_schedule, device):
        self.timesteps = timesteps
        self.noise_schedule = noise_schedule
        self.betas = noise_schedule(timesteps)

    def __call__(self, x):
        return self.sample_noise(x)
    
    def sample_noise(self, x):
        pass


class CosineScheduler():
    """Scheduler for discrete temporal values in Diffusion Model training.
    """

    def __init__(self, timesteps, s=0.008, device=None):
        self.timesteps = timesteps
        time = torch.linspace(0, timesteps, timesteps + 1)
        alphas = alphas = torch.cos((time / timesteps + s) / (1 + s) * (np.pi / 2))**2
        alphas = alphas / alphas[0]
        alphas = alphas[1:] / alphas[:-1]
        self.betas = torch.clip(1 - alphas, 0, 0.999)

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_inv_sqrt = torch.sqrt(1 / self.alphas)
        self.alphas_cumprod_sqrt = torch.sqrt(self.alphas_cumprod)
        self.alphas_cumprod_min_sqrt = torch.sqrt(1 - self.alphas_cumprod)
        
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        self.posterior_variance = self.betas * (1 - alphas_cumprod_prev) / (1 - self.alphas_cumprod)
    
    def __call__(self, img):
        """Noise image based on schedule."""
        return self.sample_noise(img)
    
    def sample_noise(self, img):
        """Sample noise from schedule."""
        pass

    
    def _obtain(self, timestep, source, target_shape):
        """Obtain values from target in timestep index for batches.
        Based on the extract function from: https://huggingface.co/blog/annotated-diffusion.
        """
        batch_size = timestep.shape[0]
        values = source.gather(-1, timestep.cpu())
        return values.reshape(batch_size, *((1,) * (len(target_shape) - 1))).to(timestep.device)
    
