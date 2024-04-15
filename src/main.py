from train import train_diffusion
from unet import UNet
from scheduler import CosineScheduler
from dataloader import get_cifar_loaders

"""
Sources:
- https://huggingface.co/blog/annotated-diffusion
- https://github.com/dome272/Diffusion-Models-pytorch
- https://github.com/dome272/Diffusion-Models-pytorch
"""

timesteps = 500

def main():
    model = UNet(out_channels=3)
    train_loader, val_loader = get_cifar_loaders(batch_size=128, root="D:\datasets\cifar100")
    train_diffusion(
        model=model,
        scheduler=CosineScheduler(timesteps=timesteps),
        train_loader=train_loader,
        val_loader=val_loader,
        early_stopping=500,
        learning_rate=1e-3,
        timesteps=timesteps,
        epochs=1000,
        log_path="C:\\Users\\matti\\OneDrive\\Documents\\Universiteit\\Ms\\Y1\\Q3\\DL\\Project\\DL_Diffusion\\logs\\log.csv",
        save_path="C:\\Users\\matti\\OneDrive\\Documents\\Universiteit\\Ms\\Y1\\Q3\\DL\\Project\\DL_Diffusion\\models\\model.pt",
    )
    
if __name__ == "__main__":
    main()