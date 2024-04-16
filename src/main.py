from train import train_diffusion
from unet import UNet
from scheduler import CosineScheduler
from dataloader import get_data_loaders
from model import SimpleUnet

"""
Sources:
- https://huggingface.co/blog/annotated-diffusion
- https://github.com/dome272/Diffusion-Models-pytorch
- https://github.com/dome272/Diffusion-Models-pytorch
"""

def main():
    model = UNet(out_channels=3)
    train_loader, val_loader, test_loader = get_data_loaders(
        "/root/DL_Diffusion/dataset/test2014",
        "/root/DL_Diffusion/dataset/val2014/",
        "/root/DL_Diffusion/dataset/test2014/",
        batch_size=512,
        )
    train_diffusion(
        model=model,
        scheduler=CosineScheduler(timesteps=1000),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        log_path="/root/DL_Diffusion/results/log.csv",
        save_path="/root/DL_Diffusion/results/model.pt",
    )
    
if __name__ == "__main__":
    main()