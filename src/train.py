import torch
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataloader import get_data_loaders
from noise import CosineScheduler



def train_diffusion(
        model,
        scheduler,
        train_loader,
        val_loader,
        test_loader=None,
        epochs=100,
        early_stopping=10,
        optimizer=Adam,
        learning_rate=1e-3,
        weight_decay=0,
        device="cuda",
        log_path=None,
        save_path=None,
):
    model.to(device)
    optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_val_loss = float("inf")
    early_stopping_counter = 0

    writer = SummaryWriter()

    if log_path is not None:
        with open(log_path, "w") as log_file:
            log_file.write("epoch,train_loss,val_loss\n")

    num_training_steps = epochs * len(train_loader) 
    progress_bar = tqdm(range(num_training_steps))
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for img, time in train_loader:
            img, noise = scheduler(img, time)
            
            img = img.to(device)
            time = time.to(device)
            noise = noise.to(device)
            optimizer.zero_grad()
            loss = mse_loss(model(img, time), img)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch + 1}")
            progress_bar.update(1)

        train_loss /= len(train_loader)
        writer.add_scalar('training loss',
            train_loss,
            epoch + 1
        )

        with torch.no_grad():
            model.eval()
            val_loss = 0
            
            progress_bar_val = tqdm(range(val_loader))
            for img, time in val_loader:
                img, noise = scheduler(img, time)
                
                img = img.to(device)
                time = time.to(device)
                noise = noise.to(device)

                loss = mse_loss(model(img, time), img)
                val_loss += loss.item()
                progress_bar_val.update(1)

            progress_bar_val.close()
            val_loss /= len(val_loader)
            writer.add_scalar('validation loss',
                val_loss,
                epoch + 1
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping:
                    print(f'--- Early Stop @ {epoch + 1} ---')
                    break

        if log_path is not None:
            with open(log_path, "a") as log_file:
                log_file.write(f"{epoch + 1},{train_loss},{val_loss}\n")
        
        print(f'Epoch: {epoch + 1}')
        print(f'Train Loss: {train_loss}')
        print(f'Validation Loss: {val_loss}', end='\n\n')

    if test_loader is not None:
        with torch.no_grad():
            model.eval()
            test_loss = 0

            progress_bar_test = tqdm(range(test_loader))

            for img, time in test_loader:
                img, noise = scheduler(img, time)
                
                img = img.to(device)
                time = time.to(device)
                noise = noise.to(device)
                loss = mse_loss(model(img, time), img)
                test_loss += loss.item()
                progress_bar_test.update(1)
            
            progress_bar_test.close()
            test_loss /= len(test_loader)
            writer.add_scalar('testing loss',
                test_loss
            )
            print(f'Test Loss: {test_loss}')

    writer.flush()
    writer.close()
    