from train import train_diffusion
from unet import UNet
from noise import CosineScheduler
from dataloader import get_data_loaders


def main():
    model = UNet(in_channels=3, out_channels=3)
    train_loader, val_loader, test_loader = get_data_loaders(
        "D:\\datasets\\coco_2014\\train\\data",
        "D:\\datasets\\coco_2014\\validation\\data",
        "D:\\datasets\\coco_2014\\test\\data",
        batch_size=64,
        )
    train_diffusion(
        model=model,
        scheduler=CosineScheduler(timesteps=1000),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        log_path="C:\\Users\\matti\\OneDrive\\Documents\\Universiteit\\Ms\\Y1\\Q3\\DL\\Project\\DL_Diffusion\\logs\\log.csv",
        save_path="C:\\Users\\matti\\OneDrive\\Documents\\Universiteit\\Ms\\Y1\\Q3\\DL\\Project\\DL_Diffusion\\models\\model.pt",
    )
    
if __name__ == "__main__":
    main()