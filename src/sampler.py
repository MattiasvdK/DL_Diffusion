import torch
import numpy as np
from noise import NoiseSampler


class DiffusionSampler(NoiseSampler):
    def __init__(self, model, timesteps, s=0.008):
        super().__init__(timesteps, s)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def __call__(self, shape=(4, 3, 256, 256), classes=100):
        # Add classes to shape
        shape = (classes, shape[0], shape[1], shape[2], shape[3])
        
        img = torch.randn(shape, device=self.device)
        
        batches = shape[1]
        imgs = []
        with torch.no_grad():
            for cls in range(classes):
                print(f'Generating samples for class {cls}')
                cls_tensor = torch.tensor(
                    np.eye(classes)[cls], dtype=torch.long, device=self.device
                ).unsqueeze(0).repeat(batches, 1)

                for time in reversed(range(self.timesteps)):
                    img[cls] = self._sample_model(
                        img[cls], torch.full((batches,), time, dtype=torch.long, device=self.device),
                        time, cls_tensor
                    )
                imgs.append(img[cls].cpu().numpy())
        return np.array(imgs)

    def _sample_model(self, img, timestep, time_idx, cls):
        
        betas_t = self._obtain(self.betas, timestep, img.shape)
        alpha_sqrt_minus_t = self._obtain(
            self.alphas_cumprod_min_sqrt, timestep, img.shape
        )
        alpha_recip_sqrt_t = self._obtain(
            self.alphas_recip_sqrt, timestep, img.shape
        )

        means = self.model(img, cls, timestep)
        model_mean = alpha_recip_sqrt_t * (
            img - betas_t * means / alpha_sqrt_minus_t
        )
        if time_idx == 0:
            noise = torch.zeros_like(img).to(self.device)
        else:
            noise = self._gaussian_noise(img.shape).to(self.device)

        posterior_variance_t = self._obtain(
            self.posterior_variance, timestep, img.shape
        )
        
        img = model_mean + torch.sqrt(posterior_variance_t) * noise

        img = (img.clamp(-1, 1))
        return img



