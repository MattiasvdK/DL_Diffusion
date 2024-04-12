from train import train_diffusion
from unet import UNet
from noise import CosineScheduler
from dataloader import get_data_loaders


def main():
    model = UNet(in_channels=3, out_channels=3)
    train_loader, val_loader, test_loader = get_data_loaders(
        "$TMPDIR/dataset/train2014/",
        "$TMPDIR/dataset/val2014/",
        "$TMPDIR/dataset/test2014/",
        batch_size=256,
        )
    train_diffusion(
        model=model,
        scheduler=CosineScheduler(timesteps=1000),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        early_stopping=5,
        log_path="$TMPDIR/results/log",
        save_path="$TMPDIR/results/model",
    )
    
if __name__ == "__main__":
    main()