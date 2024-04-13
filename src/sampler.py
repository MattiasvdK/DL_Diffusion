import torch
import numpy as np
from noise import NoiseSampler


class DiffusionSampler(NoiseSampler):
    def __init__(self, model, timesteps, s=0.008):
        super().__init__(timesteps, s)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def __call__(self, shape=(4, 3, 256, 256)):
        img = torch.randn(shape, device=self.device)

        batches = shape[0]
        imgs = []
        with torch.no_grad():
            for time in reversed(range(self.timesteps)):
                img = self._sample_model(
                    img, torch.full((batches,), time, dtype=torch.long, device=self.device), time
                )
                imgs.append(img.cpu().detach().numpy())
        return np.array(imgs)

    def _sample_model(self, img, timestep, time_idx):
        
        betas_t = self._obtain(self.betas, timestep, img.shape)
        alpha_sqrt_minus_t = self._obtain(
            self.alphas_cumprod_min_sqrt, timestep, img.shape
        )
        alpha_recip_sqrt_t = self._obtain(
            self.alphas_recip_sqrt, timestep, img.shape
        )

        means = self.model(img, time_idx)
        model_mean = alpha_recip_sqrt_t * (
            img - betas_t * means / alpha_sqrt_minus_t
        )
        if time_idx == 0:
            return model_mean

        posterior_variance_t = self._obtain(
            self.posterior_variance, timestep, img.shape
        )
        noise = self._gaussian_noise(img.shape).to(self.device)

        return model_mean + torch.sqrt(posterior_variance_t) * noise

