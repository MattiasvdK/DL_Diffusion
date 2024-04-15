import torch
import numpy as np
from torch.nn.functional import mse_loss
from torch.optim import Adam

from dataloader import get_data_loaders
from scheduler import CosineScheduler



def train_diffusion(
        model,
        scheduler,
        train_loader,
        val_loader,
        test_loader=None,
        timesteps=1000,
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

    if log_path is not None:
        with open(log_path, "w") as log_file:
            log_file.write("epoch,train_loss,val_loss\n")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for img, label in train_loader:
            time = get_time(timesteps, len(img))
            
            img, noise = scheduler(img, time)
            
            img = img.to(device)
            time = time.to(device)
            noise = noise.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            loss = mse_loss(model(img, label, time), noise)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        with torch.no_grad():
            model.eval()
            val_loss = 0
            for img, label in val_loader:
                time = get_time(timesteps, len(img))
                img, noise = scheduler(img, time)
                
                img = img.to(device)
                time = time.to(device)
                noise = noise.to(device)

                loss = mse_loss(model(img, label, time), noise)
                val_loss += loss.item()

            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping:
                    print(f'--- Early Stop @ {epoch} ---')
                    break

        if log_path is not None:
            with open(log_path, "a") as log_file:
                log_file.write(f"{epoch},{train_loss},{val_loss}\n")
        
        print(f'Epoch: {epoch}')
        print(f'Train Loss: {train_loss}')
        print(f'Validation Loss: {val_loss}', end='\n\n')

    if test_loader is not None:
        with torch.no_grad():
            model.eval()
            test_loss = 0
            for img, time in test_loader:
                img, noise = scheduler(img, time)
                
                img = img.to(device)
                time = time.to(device)
                noise = noise.to(device)
                
                loss = mse_loss(model(img, time), noise)
                test_loss += loss.item()
            
            test_loss /= len(test_loader)
            print(f'Test Loss: {test_loss}')
    

def get_time(timesteps, batch_size):
    return torch.randint(0, timesteps, (batch_size,))
