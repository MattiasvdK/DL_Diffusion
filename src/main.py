from train import train_diffusion
from unet import UNet
from noise import CosineScheduler
from dataloader import get_data_loaders
from model import SimpleUnet


def main():
    model = SimpleUnet()
    # model = UNet(in_channels=3, out_channels=3)
    train_loader, val_loader, test_loader = get_data_loaders(
        "/tmp/dataset/train2014/",
        "/tmp/dataset/val2014/",
        "/tmp/dataset/test2014/",
        batch_size=64,
        )
    train_diffusion(
        model=model,
        scheduler=CosineScheduler(timesteps=500),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        early_stopping=5,
        log_path="/tmp/results/log.csv",
        save_path="/tmp/results/model.pt",
    )
    
if __name__ == "__main__":
    main()